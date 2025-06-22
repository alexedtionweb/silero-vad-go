package main

import (
	"encoding/binary"
	"flag"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/go-audio/wav"
	"github.com/streamer45/silero-vad-go/speech"
)

func main() {
	const (
		sampleRate = 16000
		frameSize  = 512 // For 16kHz
	)

	modelPath := flag.String("model", "testfiles/silero_vad.onnx", "Path to Silero VAD ONNX model")
	flag.Parse()

	if flag.NArg() != 1 {
		log.Fatalf("Usage: %s [--model path/to/model.onnx] path/to/audio.wav|pcm", os.Args[0])
	}
	audioFile := flag.Arg(0)

	detector, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:  *modelPath,
		SampleRate: sampleRate,
		Threshold:  0.5,
	})
	if err != nil {
		log.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Destroy()

	var frameChan <-chan []float32
	ext := filepath.Ext(audioFile)
	switch ext {
	case ".pcm":
		frameChan, err = ReadPCMAndStream(audioFile, sampleRate)
		if err != nil {
			log.Fatalf("Failed to read PCM: %v", err)
		}
	case ".wav":
		frameChan, err = ReadWAVAndStream(audioFile, sampleRate)
		if err != nil {
			log.Fatalf("Failed to read WAV: %v", err)
		}
	default:
		log.Fatalf("Unsupported file extension '%s'. Only .pcm and .wav are supported.", ext)
	}

	for frame := range frameChan {
		event, err := detector.DetectStreamFrame(frame)
		if err != nil {
			if err.Error() == "unexpected speech end" {
				log.Printf("unexpected speech end, resetting detector state")
				detector.Reset()
				continue
			} else {
				log.Printf("failed to process VAD: %v", err)
				continue
			}
		}
		if event != nil {
			now := time.Now().Format("2006/01/02 15:04:05")
			if event.IsStart {
				log.Printf("%s speech starts at %.2fs", now, float64(event.StartSample)/float64(sampleRate))
			}
			if event.IsEnd {
				log.Printf("%s speech ends at %.2fs", now, float64(event.EndSample)/float64(sampleRate))
			}
		}
	}
}

func ReadPCMAndStream(filePath string, sampleRate int) (<-chan []float32, error) {
	frameSize := 512
	if sampleRate == 8000 {
		frameSize = 256
	}
	framesChan := make(chan []float32, 5)

	go func() {
		defer close(framesChan)
		file, err := os.Open(filePath)
		if err != nil {
			log.Printf("Error opening PCM file '%s': %v", filePath, err)
			return
		}
		defer file.Close()

		const bytesPerSample = 4 // float32 = 4 bytes
		byteBuffer := make([]byte, frameSize*bytesPerSample)

		for {
			n, err := io.ReadFull(file, byteBuffer)
			if err != nil {
				if err == io.EOF || err == io.ErrUnexpectedEOF {
					break
				}
				log.Printf("Error reading from PCM file: %v", err)
				return
			}
			if n < frameSize*bytesPerSample {
				break
			}

			floatFrame := make([]float32, frameSize)
			for i := 0; i < frameSize; i++ {
				floatFrame[i] = math.Float32frombits(binary.LittleEndian.Uint32(byteBuffer[i*bytesPerSample : (i+1)*bytesPerSample]))
			}
			framesChan <- floatFrame
		}
	}()
	return framesChan, nil
}

func ReadWAVAndStream(filePath string, sampleRate int) (<-chan []float32, error) {
	frameSize := 512
	if sampleRate == 8000 {
		frameSize = 256
	}
	framesChan := make(chan []float32, 5)

	go func() {
		defer close(framesChan)
		f, err := os.Open(filePath)
		if err != nil {
			log.Printf("Error opening WAV file '%s': %v", filePath, err)
			return
		}
		defer f.Close()

		dec := wav.NewDecoder(f)
		if !dec.IsValidFile() {
			log.Printf("Invalid WAV file")
			return
		}

		buf, err := dec.FullPCMBuffer()
		if err != nil {
			log.Printf("Failed to decode WAV buffer: %v", err)
			return
		}

		floatBuf := buf.AsFloat32Buffer()
		data := floatBuf.Data
		numFrames := len(data) / frameSize

		for i := 0; i < numFrames; i++ {
			start := i * frameSize
			end := start + frameSize
			if end > len(data) {
				break
			}
			frame := make([]float32, frameSize)
			copy(frame, data[start:end])
			framesChan <- frame
		}
	}()
	return framesChan, nil
}
