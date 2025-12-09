// stream-denoiser.cc
// Real-time streaming DFSMN denoiser implementation

#include "stream-denoiser.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>

#include "onnxruntime_cxx_api.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "kaldi-native-fbank/csrc/istft.h"

namespace dfsmn {

std::string StreamDenoiserConfig::ToString() const {
  std::ostringstream os;
  os << "StreamDenoiserConfig:\n";
  os << "  model_path: " << model_path << "\n";
  os << "  num_threads: " << num_threads << "\n";
  os << "  sample_rate: " << sample_rate << "\n";
  os << "  chunk_size_ms: " << chunk_size_ms << "\n";
  os << "  lookahead_ms: " << lookahead_ms << "\n";
  os << "  n_fft: " << n_fft << "\n";
  os << "  hop_length: " << hop_length << "\n";
  os << "  latency: " << (n_fft * 1000.0f / sample_rate) << " ms\n";
  return os.str();
}

class StreamDenoiser::Impl {
 public:
  explicit Impl(const StreamDenoiserConfig& config) : config_(config) {}

  bool Init() {
    try {
      InitOnnxSession();
      InitFbank();
      InitStft();
      Reset();
      initialized_ = true;
      return true;
    } catch (const std::exception& e) {
      std::cerr << "StreamDenoiser initialization failed: " << e.what() << "\n";
      return false;
    }
  }

  void Reset() {
    input_buffer_.clear();
    output_buffer_.clear();
    overlap_buffer_.assign(config_.n_fft - config_.hop_length, 0.0f);
    frames_processed_ = 0;
    samples_output_ = 0;

    // Reset fbank state
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
  }

  int32_t Process(const float* samples, int32_t n, float* output) {
    if (!initialized_) {
      std::cerr << "StreamDenoiser not initialized!\n";
      return 0;
    }

    // Add samples to input buffer
    for (int32_t i = 0; i < n; ++i) {
      input_buffer_.push_back(samples[i]);
    }

    int32_t output_count = 0;

    // Process when we have enough samples for at least one frame
    while (input_buffer_.size() >= static_cast<size_t>(config_.n_fft)) {
      // Extract frame for processing
      std::vector<float> frame(input_buffer_.begin(),
                               input_buffer_.begin() + config_.n_fft);

      // Process single frame
      ProcessFrame(frame);

      // Move hop_length samples from output buffer to output
      int32_t available = std::min(static_cast<int32_t>(output_buffer_.size()),
                                   config_.hop_length);
      for (int32_t i = 0; i < available; ++i) {
        output[output_count++] = output_buffer_.front();
        output_buffer_.pop_front();
      }

      // Advance input buffer by hop_length
      input_buffer_.erase(input_buffer_.begin(),
                          input_buffer_.begin() + config_.hop_length);
    }

    return output_count;
  }

  void ProcessWithCallback(const float* samples, int32_t n,
                           const AudioOutputCallback& callback) {
    // Temporary buffer for output
    std::vector<float> output(n + config_.n_fft);
    int32_t output_count = Process(samples, n, output.data());

    if (output_count > 0 && callback) {
      callback(output.data(), output_count);
    }
  }

  int32_t Flush(float* output, int32_t max_samples) {
    if (!initialized_) return 0;

    // Pad remaining input with zeros to flush
    int32_t remaining = input_buffer_.size();
    if (remaining > 0) {
      // Pad to complete the last frame
      int32_t pad_size = config_.n_fft - remaining;
      for (int32_t i = 0; i < pad_size; ++i) {
        input_buffer_.push_back(0.0f);
      }

      // Process remaining
      std::vector<float> frame(input_buffer_.begin(), input_buffer_.end());
      ProcessFrame(frame);
      input_buffer_.clear();
    }

    // Output remaining samples
    int32_t output_count = std::min(static_cast<int32_t>(output_buffer_.size()),
                                    max_samples);
    for (int32_t i = 0; i < output_count; ++i) {
      output[i] = output_buffer_.front();
      output_buffer_.pop_front();
    }

    return output_count;
  }

  int32_t GetLatencySamples() const {
    // Latency = n_fft (one full frame needed before output)
    return config_.n_fft;
  }

  float GetLatencyMs() const {
    return GetLatencySamples() * 1000.0f / config_.sample_rate;
  }

  int32_t GetChunkSizeSamples() const {
    return config_.hop_length;  // Process hop_length samples at a time
  }

  bool IsInitialized() const { return initialized_; }

 private:
  void InitOnnxSession() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "stream-denoiser");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config_.num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(),
                                               session_options);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name.get();

    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = output_name.get();

    std::cerr << "ONNX model loaded: " << config_.model_path << "\n";
  }

  void InitFbank() {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = config_.sample_rate;
    opts.frame_opts.frame_length_ms = config_.frame_length_ms;
    opts.frame_opts.frame_shift_ms = config_.frame_shift_ms;
    opts.frame_opts.dither = config_.dither;
    opts.frame_opts.snip_edges = false;
    opts.frame_opts.window_type = "hamming";
    opts.mel_opts.num_bins = config_.num_mel_bins;
    opts.mel_opts.low_freq = 20.0f;
    opts.mel_opts.high_freq = config_.sample_rate / 2.0f;

    fbank_opts_ = opts;
    fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);
  }

  void InitStft() {
    knf::StftConfig stft_config;
    stft_config.n_fft = config_.n_fft;
    stft_config.hop_length = config_.hop_length;
    stft_config.win_length = config_.win_length;
    stft_config.window_type = "hamming";
    stft_config.center = false;

    stft_ = std::make_unique<knf::Stft>(stft_config);
    istft_ = std::make_unique<knf::IStft>(stft_config);
  }

  void ProcessFrame(const std::vector<float>& frame) {
    // Scale to int16 range
    std::vector<float> scaled(frame.size());
    for (size_t i = 0; i < frame.size(); ++i) {
      scaled[i] = frame[i] * 32768.0f;
    }

    // Extract fbank features for this frame
    fbank_->AcceptWaveform(config_.sample_rate, scaled.data(), scaled.size());

    int32_t num_frames = fbank_->NumFramesReady() - frames_processed_;
    if (num_frames <= 0) {
      // Not enough data for a frame yet, output zeros
      for (int32_t i = 0; i < config_.hop_length; ++i) {
        output_buffer_.push_back(0.0f);
      }
      return;
    }

    // Get fbank features
    std::vector<float> fbank_features(num_frames * config_.num_mel_bins);
    for (int32_t i = 0; i < num_frames; ++i) {
      const float* feat = fbank_->GetFrame(frames_processed_ + i);
      std::copy(feat, feat + config_.num_mel_bins,
                fbank_features.data() + i * config_.num_mel_bins);
    }
    frames_processed_ += num_frames;

    // Run inference
    std::vector<float> mask = RunInference(fbank_features, num_frames);

    // Compute STFT on scaled audio
    knf::StftResult stft_result = stft_->Compute(scaled.data(), scaled.size());

    // Apply mask
    int32_t freq_bins = config_.n_fft / 2 + 1;
    int32_t min_frames = std::min(stft_result.num_frames, num_frames);

    knf::StftResult masked_stft;
    masked_stft.num_frames = min_frames;
    masked_stft.real.resize(min_frames * freq_bins);
    masked_stft.imag.resize(min_frames * freq_bins);

    for (int32_t f = 0; f < min_frames; ++f) {
      for (int32_t b = 0; b < freq_bins; ++b) {
        int32_t idx = f * freq_bins + b;
        float m = (idx < static_cast<int32_t>(mask.size())) ? mask[idx] : 1.0f;
        masked_stft.real[idx] = stft_result.real[idx] * m;
        masked_stft.imag[idx] = stft_result.imag[idx] * m;
      }
    }

    // ISTFT reconstruction
    std::vector<float> denoised = istft_->Compute(masked_stft);

    // Overlap-add with previous frame
    int32_t overlap_size = overlap_buffer_.size();
    for (int32_t i = 0; i < overlap_size && i < static_cast<int32_t>(denoised.size()); ++i) {
      denoised[i] += overlap_buffer_[i];
    }

    // Output hop_length samples, save rest for overlap
    int32_t output_samples = std::min(config_.hop_length,
                                      static_cast<int32_t>(denoised.size()));
    for (int32_t i = 0; i < output_samples; ++i) {
      float sample = std::clamp(denoised[i] / 32768.0f, -1.0f, 1.0f);
      output_buffer_.push_back(sample);
    }

    // Update overlap buffer
    overlap_buffer_.clear();
    for (size_t i = output_samples; i < denoised.size(); ++i) {
      overlap_buffer_.push_back(denoised[i]);
    }
    // Pad if needed
    while (overlap_buffer_.size() < static_cast<size_t>(config_.n_fft - config_.hop_length)) {
      overlap_buffer_.push_back(0.0f);
    }

    samples_output_ += output_samples;
  }

  std::vector<float> RunInference(const std::vector<float>& fbank,
                                   int32_t num_frames) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> input_shape{1, num_frames, config_.num_mel_bins};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(fbank.data()), fbank.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    auto& output_tensor = outputs[0];
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    const float* output_data = output_tensor.GetTensorData<float>();

    return std::vector<float>(output_data, output_data + output_size);
  }

  StreamDenoiserConfig config_;
  bool initialized_ = false;

  // ONNX Runtime
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;

  // Feature extraction
  knf::FbankOptions fbank_opts_;
  std::unique_ptr<knf::OnlineFbank> fbank_;
  std::unique_ptr<knf::Stft> stft_;
  std::unique_ptr<knf::IStft> istft_;

  // Streaming buffers
  std::deque<float> input_buffer_;
  std::deque<float> output_buffer_;
  std::vector<float> overlap_buffer_;

  // State tracking
  int32_t frames_processed_ = 0;
  int64_t samples_output_ = 0;
};

// StreamDenoiser public interface implementation

StreamDenoiser::StreamDenoiser(const StreamDenoiserConfig& config)
    : impl_(std::make_unique<Impl>(config)), config_(config) {}

StreamDenoiser::~StreamDenoiser() = default;

StreamDenoiser::StreamDenoiser(StreamDenoiser&&) noexcept = default;
StreamDenoiser& StreamDenoiser::operator=(StreamDenoiser&&) noexcept = default;

bool StreamDenoiser::Init() { return impl_->Init(); }

void StreamDenoiser::Reset() { impl_->Reset(); }

int32_t StreamDenoiser::Process(const float* samples, int32_t n, float* output) {
  return impl_->Process(samples, n, output);
}

void StreamDenoiser::ProcessWithCallback(const float* samples, int32_t n,
                                          const AudioOutputCallback& callback) {
  impl_->ProcessWithCallback(samples, n, callback);
}

int32_t StreamDenoiser::Flush(float* output, int32_t max_samples) {
  return impl_->Flush(output, max_samples);
}

int32_t StreamDenoiser::GetLatencySamples() const {
  return impl_->GetLatencySamples();
}

float StreamDenoiser::GetLatencyMs() const {
  return impl_->GetLatencyMs();
}

int32_t StreamDenoiser::GetChunkSizeSamples() const {
  return impl_->GetChunkSizeSamples();
}

bool StreamDenoiser::IsInitialized() const {
  return impl_->IsInitialized();
}

}  // namespace dfsmn
