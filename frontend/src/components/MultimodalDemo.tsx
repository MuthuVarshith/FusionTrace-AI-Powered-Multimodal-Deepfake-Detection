
import React, { useState } from 'react';
import { FileCheck, Play, Slice as VoiceIcon } from 'lucide-react';

const MultimodalDemo: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState({ audio: false, image: false, video: false });
  const [demoResult, setDemoResult] = useState<{ audio: string | null; image: string | null; video: string | null }>({
    audio: null,
    image: null,
    video: null,
  });

  const handleDemoDetect = async (type: 'audio' | 'image') => {
    setIsProcessing((prev) => ({ ...prev, [type]: true }));
    try {
      const formData = new FormData();
      const response = await fetch(`http://localhost:8000/${type}/detect`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setDemoResult((prev) => ({ ...prev, [type]: result.prediction }));
    } catch (error) {
      setDemoResult((prev) => ({ ...prev, [type]: 'Error' }));
    } finally {
      setIsProcessing((prev) => ({ ...prev, [type]: false }));
    }
  };

  return (
    <div className="max-w-6xl mx-auto bg-gray-800 rounded-2xl p-8 shadow-xl mb-24">
      <div className="text-center mb-12">
        <h3 className="text-3xl font-bold text-white mb-4">Experience Multimodal Detection</h3>
        <p className="text-gray-400">Test our AI's ability to detect deepfakes across different media types</p>
      </div>
      <div className="grid md:grid-cols-3 gap-8">
        <div className="bg-gray-700 p-6 rounded-xl">
          <div className="text-center mb-6">
            <VoiceIcon className="h-12 w-12 text-purple-500 mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-white mb-2">Audio Detection</h4>
            <p className="text-gray-400 text-sm">Detect AI-generated voices and audio deepfakes</p>
          </div>
          <div className="flex justify-center mb-4">
            <audio controls className="w-full" src="https://example.com/demo-audio.mp3">
              Your browser does not support the audio element.
            </audio>
          </div>
          <button
            onClick={() => handleDemoDetect('audio')}
            disabled={isProcessing.audio}
            className={`w-full py-2 rounded-lg transition-colors ${isProcessing.audio ? 'bg-gray-600 cursor-not-allowed' : 'bg-purple-500 hover:bg-purple-400 text-white'}`}
          >
            {isProcessing.audio ? 'Processing...' : 'Analyze Audio'}
          </button>
          {demoResult.audio && (
            <p className={`text-center mt-4 ${demoResult.audio === 'Real' ? 'text-green-400' : 'text-red-400'}`}>
              Result: {demoResult.audio}
            </p>
          )}
        </div>
        <div className="bg-gray-700 p-6 rounded-xl">
          <div className="text-center mb-6">
            <FileCheck className="h-12 w-12 text-purple-500 mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-white mb-2">Image Detection</h4>
            <p className="text-gray-400 text-sm">Identify AI-generated and manipulated images</p>
          </div>
          <div className="mb-4">
            <img
              src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=800&q=80"
              alt="Demo face"
              className="w-full h-48 object-cover rounded-lg"
            />
          </div>
          <button
            onClick={() => handleDemoDetect('image')}
            disabled={isProcessing.image}
            className={`w-full py-2 rounded-lg transition-colors ${isProcessing.image ? 'bg-gray-600 cursor-not-allowed' : 'bg-purple-500 hover:bg-purple-400 text-white'}`}
          >
            {isProcessing.image ? 'Processing...' : 'Analyze Image'}
          </button>
          {demoResult.image && (
            <p className={`text-center mt-4 ${demoResult.image === 'Real' ? 'text-green-400' : 'text-red-400'}`}>
              Result: {demoResult.image}
            </p>
          )}
        </div>
        <div className="bg-gray-700 p-6 rounded-xl">
          <div className="text-center mb-6">
            <Play className="h-12 w-12 text-purple-500 mx-auto mb-4" />
            <h4 className="text-xl font-semibold text-white mb-2">Video Detection</h4>
            <p className="text-gray-400 text-sm">Spot deepfake videos and manipulated content (Coming Soon)</p>
          </div>
          <div className="mb-4">
            <video className="w-full h-48 object-cover rounded-lg" controls>
              <source src="https://example.com/demo-video.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
          <button disabled className="w-full py-2 rounded-lg bg-gray-600 cursor-not-allowed text-white">
            Analyze Video (Not Available)
          </button>
        </div>
      </div>
    </div>
  );
};

export default MultimodalDemo;
