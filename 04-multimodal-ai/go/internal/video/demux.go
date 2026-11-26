package video

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"

	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/contract"
)

type DemuxResult struct {
	Frames   [][]byte
	AudioWAV []byte
}

// Demux splits raw video bytes into sampled RGB PNG frames and a 16kHz mono PCM WAV track via ffmpeg.
func Demux(ctx context.Context, videoBytes []byte) (*DemuxResult, error) {
	tmpDir, err := os.MkdirTemp("", "multimodal-demux-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	inPath := filepath.Join(tmpDir, "input.mp4")
	if err := os.WriteFile(inPath, videoBytes, 0o600); err != nil {
		return nil, fmt.Errorf("write temp video: %w", err)
	}

	// 1. Sample frames at contract.VideoSampleFPS, scaled directly to VisionWidth x VisionHeight
	framePattern := filepath.Join(tmpDir, "frame_%03d.png")
	vf := fmt.Sprintf("fps=%d,scale=%d:%d", contract.VideoSampleFPS, contract.VisionWidth, contract.VisionHeight)
	cmdFrames := exec.CommandContext(ctx, "ffmpeg", "-y", "-i", inPath, "-vf", vf, "-frames:v", fmt.Sprintf("%d", contract.VideoMaxFrames), framePattern)
	if out, err := cmdFrames.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("ffmpeg extract frames: %w (%s)", err, string(out))
	}

	paths, err := filepath.Glob(filepath.Join(tmpDir, "frame_*.png"))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)

	frames := make([][]byte, 0, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		frames = append(frames, b)
	}

	// 2. Extract 16kHz mono 16-bit PCM WAV (optional: some clips may lack an audio stream)
	wavPath := filepath.Join(tmpDir, "audio.wav")
	cmdAudio := exec.CommandContext(ctx, "ffmpeg", "-y", "-i", inPath, "-vn", "-ac", fmt.Sprintf("%d", contract.AudioChannels), "-ar", fmt.Sprintf("%d", contract.AudioSampleRate), "-f", "wav", wavPath)
	var audioWAV []byte
	if err := cmdAudio.Run(); err == nil {
		audioWAV, _ = os.ReadFile(wavPath)
	}

	return &DemuxResult{Frames: frames, AudioWAV: audioWAV}, nil
}
