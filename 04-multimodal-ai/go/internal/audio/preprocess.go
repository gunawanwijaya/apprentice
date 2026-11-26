package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/contract"
)

// ParseWAV16kMono extracts normalized float32 samples [-1, 1] from a 16-bit mono PCM WAV payload.
func ParseWAV16kMono(data []byte) ([]float32, error) {
	if len(data) < 44 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, errors.New("invalid WAV header")
	}
	pos := 12
	var pcm []byte
	for pos+8 <= len(data) {
		chunkID := string(data[pos : pos+4])
		chunkSize := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		pos += 8
		if pos+chunkSize > len(data) {
			return nil, fmt.Errorf("truncated WAV chunk %q", chunkID)
		}
		if chunkID == "data" {
			pcm = data[pos : pos+chunkSize]
			break
		}
		pos += chunkSize
		if chunkSize%2 == 1 {
			pos++
		}
	}
	if len(pcm) == 0 {
		return nil, errors.New("no PCM data chunk found in WAV")
	}

	n := len(pcm) / 2
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		s := int16(binary.LittleEndian.Uint16(pcm[2*i : 2*i+2]))
		samples[i] = float32(s) / 32768.0
	}
	return samples, nil
}

// ExtractFeatures produces a flattened [num_frames, 2] float32 slice (log_rms, zcr) matching Python 1:1.
func ExtractFeatures(wavBytes []byte) ([]float32, error) {
	samples, err := ParseWAV16kMono(wavBytes)
	if err != nil {
		return nil, err
	}

	frameLen := (contract.AudioSampleRate * contract.AudioFrameMs) / 1000
	hopLen := (contract.AudioSampleRate * contract.AudioHopMs) / 1000
	numFrames := contract.AudioNumFrames

	out := make([]float32, numFrames*2)
	frame := make([]float32, frameLen)

	for i := 0; i < numFrames; i++ {
		start := i * hopLen
		for j := 0; j < frameLen; j++ {
			if start+j < len(samples) {
				frame[j] = samples[start+j]
			} else {
				frame[j] = 0
			}
		}

		var sumSq float64
		var crossings int
		for j := 0; j < frameLen; j++ {
			v := float64(frame[j])
			sumSq += v * v
			if j > 0 && ((frame[j] >= 0) != (frame[j-1] >= 0)) {
				crossings++
			}
		}

		rms := math.Sqrt(sumSq/float64(frameLen) + 1e-8)
		out[2*i] = float32(math.Log(rms + 1e-6))
		out[2*i+1] = float32(float64(crossings) / float64(frameLen-1))
	}
	return out, nil
}
