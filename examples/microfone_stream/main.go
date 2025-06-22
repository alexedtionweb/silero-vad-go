package main

import (
	"fmt"
	"log"

	"github.com/gordonklaus/portaudio"
	"github.com/streamer45/silero-vad-go/speech"
)

func main() {
	const (
		sampleRate = 16000
		frameSize  = 512 // Each frame is 512 samples for Silero at 16kHz
	)

	config := speech.DetectorConfig{
		ModelPath:  "testfiles/silero_vad.onnx",
		SampleRate: sampleRate,
		Threshold:  0.5,
	}

	detector, err := speech.NewDetector(config)
	if err != nil {
		log.Fatalf("Failed to create detector: %v", err)
	}
	defer detector.Destroy()

	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("Failed to init PortAudio: %v", err)
	}
	defer portaudio.Terminate()

	streamBuffer := make([]float32, frameSize)
	stream, err := portaudio.OpenDefaultStream(1, 0, float64(sampleRate), len(streamBuffer), streamBuffer)
	if err != nil {
		log.Fatalf("Failed to open stream: %v", err)
	}
	defer stream.Close()
	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start stream: %v", err)
	}

	fmt.Println("Started Recording (streaming VAD, ctrl+c to exit)")
	for {
		if err := stream.Read(); err != nil {
			log.Fatalf("Failed to read stream: %v", err)
		}

		// Streaming VAD: process each frame
		event, err := detector.DetectStreamFrame(streamBuffer)
		if err != nil {
			if err.Error() == "unexpected speech end" {
				log.Printf("Warning: %v - Resetting detector state.", err)
				detector.Reset()
				continue
			} else {
				log.Printf("Failed to process VAD: %v", err)
				continue
			}
		}
		if event != nil {
			if event.IsStart {
				fmt.Printf("Speech started at sample %d (%.2f sec)\n", event.StartSample, float64(event.StartSample)/float64(sampleRate))
			}
			if event.IsEnd {
				fmt.Printf("Speech ended at sample %d (%.2f sec)\n", event.EndSample, float64(event.EndSample)/float64(sampleRate))
			}
		}
	}
}
