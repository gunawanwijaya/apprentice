package vision

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/contract"
)

// DecodeToCHW decodes PNG/JPEG bytes, resizes if needed, and returns a [3, H, W] float32 slice.
func DecodeToCHW(data []byte) ([]float32, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	dstW, dstH := contract.VisionWidth, contract.VisionHeight
	plane := dstW * dstH
	out := make([]float32, 3*plane)

	for y := 0; y < dstH; y++ {
		sy := b.Min.Y + (y*srcH)/dstH
		for x := 0; x < dstW; x++ {
			sx := b.Min.X + (x*srcW)/dstW
			r16, g16, b16, _ := img.At(sx, sy).RGBA()
			rf := float32(r16>>8) / 255.0
			gf := float32(g16>>8) / 255.0
			bf := float32(b16>>8) / 255.0

			idx := y*dstW + x
			out[idx] = (rf - contract.VisionMean[0]) / contract.VisionStd[0]
			out[plane+idx] = (gf - contract.VisionMean[1]) / contract.VisionStd[1]
			out[2*plane+idx] = (bf - contract.VisionMean[2]) / contract.VisionStd[2]
		}
	}
	return out, nil
}

// CosineSimilarity compares two flattened CHW frame tensors for video frame deduplication.
func CosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := 0; i < len(a) && i < len(b); i++ {
		fa, fb := float64(a[i]), float64(b[i])
		dot += fa * fb
		normA += fa * fa
		normB += fb * fb
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
