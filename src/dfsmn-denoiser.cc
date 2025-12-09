// dfsmn-denoiser.cc
// DFSMN ANS Denoiser Implementation

#include "dfsmn-denoiser.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include "onnxruntime_cxx_api.h"
#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "kaldi-native-fbank/csrc/istft.h"
#include "resample.h"

namespace dfsmn {

std::string DenoiserConfig::ToString() const {
  std::ostringstream os;
  os << "DenoiserConfig:\n";
  os << "  model_path: " << model_path << "\n";
  os << "  num_threads: " << num_threads << "\n";
  os << "  sample_rate: " << sample_rate << "\n";
  os << "  n_fft: " << n_fft << "\n";
  os << "  hop_length: " << hop_length << "\n";
  os << "  win_length: " << win_length << "\n";
  os << "  num_mel_bins: " << num_mel_bins << "\n";
  os << "  frame_length_ms: " << frame_length_ms << "\n";
  os << "  frame_shift_ms: " << frame_shift_ms << "\n";
  os << "  dither: " << dither << "\n";
  return os.str();
}

class DfsmnDenoiser::Impl {
 public:
  explicit Impl(const DenoiserConfig& config) : config_(config) {
    InitOnnxSession();
    InitFbank();
    InitStft();
  }

  DenoisedAudio Run(const float* samples, int32_t n, int32_t sample_rate) const {
    // Step 1: Resample if needed
    std::vector<float> resampled;
    const float* audio_data = samples;
    int32_t audio_len = n;

    if (sample_rate != config_.sample_rate) {
      std::cerr << "Resampling from " << sample_rate << " to " << config_.sample_rate << " Hz\n";
      float min_freq = std::min<float>(sample_rate, config_.sample_rate);
      float lowpass_cutoff = 0.99f * 0.5f * min_freq;
      int32_t lowpass_filter_width = 6;

      sherpa_onnx::LinearResample resampler(sample_rate, config_.sample_rate,
                                            lowpass_cutoff, lowpass_filter_width);
      resampler.Resample(samples, n, true, &resampled);
      audio_data = resampled.data();
      audio_len = resampled.size();
    }

    // Step 2: Scale to int16 range (as in Python: audio * 32768)
    std::vector<float> audio_scaled(audio_len);
    for (int32_t i = 0; i < audio_len; ++i) {
      audio_scaled[i] = audio_data[i] * 32768.0f;
    }

    // Step 3: Extract Fbank features
    std::vector<float> fbank_features = ExtractFbank(audio_scaled);
    int32_t num_frames = fbank_features.size() / config_.num_mel_bins;
    std::cerr << "Extracted " << num_frames << " fbank frames\n";

    // Step 4: Run ONNX inference to get mask
    std::vector<float> mask = RunInference(fbank_features, num_frames);
    std::cerr << "Got mask with shape [1, " << num_frames << ", " << (config_.n_fft / 2 + 1) << "]\n";

    // Step 5: Compute STFT
    knf::StftResult stft_result = stft_->Compute(audio_scaled.data(), audio_scaled.size());
    std::cerr << "STFT frames: " << stft_result.num_frames << "\n";

    // Step 6: Apply mask to STFT
    // mask shape: [1, num_frames, n_fft/2+1] -> need to match STFT frames
    int32_t freq_bins = config_.n_fft / 2 + 1;
    int32_t min_frames = std::min(stft_result.num_frames, num_frames);

    knf::StftResult masked_stft;
    masked_stft.num_frames = min_frames;
    masked_stft.real.resize(min_frames * freq_bins);
    masked_stft.imag.resize(min_frames * freq_bins);

    for (int32_t f = 0; f < min_frames; ++f) {
      for (int32_t b = 0; b < freq_bins; ++b) {
        int32_t stft_idx = f * freq_bins + b;
        // mask is [1, frames, bins], direct indexing (no permute needed)
        int32_t mask_idx = f * freq_bins + b;
        float m = mask[mask_idx];
        masked_stft.real[stft_idx] = stft_result.real[stft_idx] * m;
        masked_stft.imag[stft_idx] = stft_result.imag[stft_idx] * m;
      }
    }

    // Step 7: ISTFT reconstruction
    std::vector<float> denoised = istft_->Compute(masked_stft);

    // Normalize back to [-1, 1]
    DenoisedAudio result;
    result.sample_rate = config_.sample_rate;
    result.samples.resize(std::min<size_t>(denoised.size(), audio_len));
    for (size_t i = 0; i < result.samples.size(); ++i) {
      result.samples[i] = std::clamp(denoised[i] / 32768.0f, -1.0f, 1.0f);
    }

    return result;
  }

 private:
  void InitOnnxSession() {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dfsmn-denoiser");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(config_.num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    input_name_ = input_name.get();

    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    output_name_ = output_name.get();

    std::cerr << "ONNX model loaded: " << config_.model_path << "\n";
    std::cerr << "  Input: " << input_name_ << "\n";
    std::cerr << "  Output: " << output_name_ << "\n";
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
    std::cerr << "Fbank initialized:\n" << opts.ToString() << "\n";
  }

  void InitStft() {
    knf::StftConfig stft_config;
    stft_config.n_fft = config_.n_fft;
    stft_config.hop_length = config_.hop_length;
    stft_config.win_length = config_.win_length;
    stft_config.window_type = "hamming";
    stft_config.center = false;  // Match Python center=False

    stft_ = std::make_unique<knf::Stft>(stft_config);
    istft_ = std::make_unique<knf::IStft>(stft_config);
    std::cerr << "STFT/ISTFT initialized:\n" << stft_config.ToString() << "\n";
  }

  std::vector<float> ExtractFbank(const std::vector<float>& audio) const {
    knf::OnlineFbank fbank(fbank_opts_);
    fbank.AcceptWaveform(config_.sample_rate, audio.data(), audio.size());
    fbank.InputFinished();

    int32_t num_frames = fbank.NumFramesReady();
    std::vector<float> features(num_frames * config_.num_mel_bins);

    for (int32_t i = 0; i < num_frames; ++i) {
      const float* frame = fbank.GetFrame(i);
      std::copy(frame, frame + config_.num_mel_bins,
                features.data() + i * config_.num_mel_bins);
    }

    return features;
  }

  std::vector<float> RunInference(const std::vector<float>& fbank,
                                   int32_t num_frames) const {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Input shape: [batch=1, num_frames, num_mel_bins=120]
    std::array<int64_t, 3> input_shape{1, num_frames, config_.num_mel_bins};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(fbank.data()), fbank.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    // Output shape: [1, num_frames, n_fft/2+1] = [1, num_frames, 961]
    auto& output_tensor = outputs[0];
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    std::cerr << "Model output shape: [" << output_shape[0] << ", "
              << output_shape[1] << ", " << output_shape[2] << "]\n";

    const float* output_data = output_tensor.GetTensorData<float>();
    return std::vector<float>(output_data, output_data + output_size);
  }

  DenoiserConfig config_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;

  knf::FbankOptions fbank_opts_;
  std::unique_ptr<knf::Stft> stft_;
  std::unique_ptr<knf::IStft> istft_;
};

DfsmnDenoiser::DfsmnDenoiser(const DenoiserConfig& config)
    : impl_(std::make_unique<Impl>(config)), config_(config) {}

DfsmnDenoiser::~DfsmnDenoiser() = default;

DenoisedAudio DfsmnDenoiser::Run(const float* samples, int32_t n,
                                  int32_t sample_rate) const {
  return impl_->Run(samples, n, sample_rate);
}

}  // namespace dfsmn
