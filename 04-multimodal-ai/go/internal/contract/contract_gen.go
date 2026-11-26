// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion = "v1.0.0"
	SchemaHash    = "559a7112982a"

	VisionWidth    = 128
	VisionHeight   = 128
	VisionChannels = 3

	AudioSampleRate = 16000
	AudioChannels   = 1
	AudioFrameMs    = 25
	AudioHopMs      = 10
	AudioNumFrames  = 100

	VideoSampleFPS      = 1
	VideoMaxFrames      = 10
	VideoDedupThreshold = 0.98
)

var (
	VisionMean = [3]float32{ 0.485, 0.456, 0.406 }
	VisionStd  = [3]float32{ 0.229, 0.224, 0.225 }

	EKYCLabels = []string{
		"clean_document",
		"blurry",
		"heavy_glare",
		"low_light",
	}
	UnboxingLabels = []string{
		"intact_parcel",
		"damaged_box",
		"empty_box",
		"blurry_frame",
	}
	AudioLabels = []string{
		"silence_or_noise",
		"speech_present",
	}
)
