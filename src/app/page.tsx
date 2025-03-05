'use client';

import {useEffect, useRef, useState, MouseEvent} from "react";
import {relu, softmax} from "@/lib/activation_functions";
import NeuralNetwork from "@/lib/NeuralNetwork";
import {decodeMnistFromUrls} from "@/lib/mnist";
import Matrix from "@/lib/Matrix";

export default function Home() {
  const [ nn, setNN ] = useState<NeuralNetwork | null>(null);
  const [ accuracy, setAccuracy ] = useState<number | null>(null);
  const [ neurons, setNeurons ] = useState(10);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ image, setImage ] = useState<number[]>(new Array(28*28).fill(0));
  const [ recognition, setRecognition ] = useState<number[] | null>(null);
  const [ since, setSince ] = useState(-1);
  const [ isDrawing, setIsDrawing ] = useState(false);
  const TIME_UNTIL_FULLY_WHITE = 10;

  const canvasDimensions = 600;
  const size = canvasDimensions / 28;

  function onMouseMove(e: MouseEvent) {
    if (!isDrawing) return;

    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;

    const relativeX = e.clientX - canvas.offsetLeft;
    const relativeY = e.clientY - canvas.offsetTop;

    const realX = relativeX / canvasDimensions * 28;
    const realY = relativeY / canvasDimensions * 28;
    const x = Math.floor(realX);
    const y = Math.floor(realY);

    function paintPixel(x: number, y: number, color: number) {
      color = Math.min(1, Math.max(0, color));

      ctx.fillStyle = `rgba(255, 255, 255, ${color})`;
      ctx.fillRect(x * size, y * size, size, size);

      // update image state
      const newImage = [...image];
      newImage[y * 28 + x] = Math.max(newImage[y * 28 + x], color);

      // also paint the surrounding pixels
      if (x > 0) {
        newImage[y * 28 + x - 1] = Math.max(newImage[y * 28 + x - 1], color);
      }
      if (x < 27) {
        newImage[y * 28 + x + 1] = Math.max(newImage[y * 28 + x + 1], color);
      }
      if (y > 0) {
        newImage[(y - 1) * 28 + x] = Math.max(newImage[(y - 1) * 28 + x], color);
      }
      if (y < 27) {
        newImage[(y + 1) * 28 + x] = Math.max(newImage[(y + 1) * 28 + x], color);
      }
      setImage(newImage);
    }

    setSince(Date.now());

    // if its close to the edges, paint the surrounding pixels
    if (realX % 1 < 0.5) {
      paintPixel(x - 1, y, (realX % 1) / 0.1);
    }

    if (realX % 1 > 0.5) {
      paintPixel(x + 1, y, (1 - realX % 1) / 0.1);
    }

    if (realY % 1 < 0.5) {
      paintPixel(x, y - 1, (realY % 1) / 0.1);
    }

    if (realY % 1 > 0.5) {
      paintPixel(x, y + 1, (1 - realY % 1) / 0.1);
    }

    paintPixel(x, y, Math.min(1, (Date.now() - since) / TIME_UNTIL_FULLY_WHITE));
  }

  useEffect(() => {
    const timeout = setInterval(() => {
      if (nn) {
        const output = nn.run(Matrix.from(image, 28*28, 1));
        setRecognition(output.data);
      }
    }, 5);

    return () => clearInterval(timeout);
  }, [ image, nn ]);

  async function train(neurons: number = 10) {
    setNN(null);
    setAccuracy(null);

    // create neural network
    const nn = new NeuralNetwork(28*28, { learningRate: 0.1 });
    nn.addLayer(neurons, relu);
    nn.addLayer(10, softmax);
    nn.randomizeParams();

    // fetch data and train
    decodeMnistFromUrls('/mnist/train_images', '/mnist/train_labels').then(entries => {
      for (const entry of entries) {
        const input = Matrix.from(entry.image, 28*28, 1);

        // normalize output
        const output = new Matrix(10, 1);
        output.set(0, entry.label, 1);

        nn.train(input, output);
      }
    }).then(() => {
      setNN(nn);

      // test the neural network
      decodeMnistFromUrls('/mnist/test_images', '/mnist/test_labels').then(entries => {
        let correct = 0;
        for (const entry of entries) {
          const input = Matrix.from(entry.image, 28*28, 1);
          const output = nn.run(input);

          let digit = -1;
          let maxConfidence = -1;
          for (let i = 0; i < 10; i++) {
            if (output.get(0, i) > maxConfidence) {
              maxConfidence = output.get(0, i);
              digit = i;
            }
          }

          if (digit === entry.label) {
            correct++;
          }
        }

        setAccuracy(correct / entries.length);
      });
    });
  }

  useEffect(() => {
    train();
  }, []);

  return (
    <div className="bg-black h-screen text-white w-screen py-8">
      <h1 className="text-2xl text-center">Neural Network Test</h1>

      <p className="text-sm text-white/40 text-center mb-8">
        {nn ? 'Neural network is trained' : 'Training neural network...'}
        {accuracy ? ` (Tested accuracy: ${Math.round(accuracy * 100)}%)` : ''}
      </p>

      <div className="mx-auto max-w-min">
        <div className="flex flex-row gap-3">
          <div className="w-min border border-white/20">
            <canvas
              className="border p-0"
              ref={canvasRef}
              width={canvasDimensions}
              height={canvasDimensions}
              onMouseMove={onMouseMove}
              onMouseDown={() => setIsDrawing(true)}
              onMouseUp={() => setIsDrawing(false)}
              onMouseOut={() => setIsDrawing(false)}
            />
          </div>

          <div className="flex-col">
            <span>Probabilities</span>

            {new Array(10).fill(0).map((_, i) => (
              <div key={i} className="flex flex-row gap-2 items-center">
                <span>{i}</span>
                <div className="w-32 border border-white/20 h-5">
                  <div style={{
                    width: `${recognition ? (recognition[i] * 100) : 0}%`,
                  }} className="h-full bg-white">
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-row py-4 gap-4 items-center">
          <button className="bg-white/10 hover:bg-white/30 cursor-pointer px-5 py-2" onClick={() => {
            const canvas = canvasRef.current!;
            const ctx = canvas.getContext('2d')!;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            setImage(new Array(28*28).fill(0));
          }}>Clear</button>

          <label className="flex flex-row items-center">Neurons (Single Hidden Layer): <input type="range" min={1} max={200} onChange={e => {
            setNeurons(parseInt(e.target.value));
          }} value={neurons} /> {neurons}</label>

          <button className="bg-white/10 hover:bg-white/30 cursor-pointer px-5 py-2" disabled={nn === null} onClick={() => train(neurons)}>Train</button>
        </div>
      </div>
    </div>
  );
}
