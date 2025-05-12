import React, { useState } from 'react';
import { Upload, AlertCircle } from 'lucide-react';

interface DetectionResult {
  prediction: 'real' | 'fake';
  confidence: number;
  saved_file_path: string;
}

const UploadDetect: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const handleDetect = async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    const endpoint = file.type.startsWith('audio/') ? '/api/audio/detect' : '/api/image/detect';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: DetectionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during detection');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div id="detect" className="mt-24 mb-24 bg-[#1A1A1A] rounded-2xl p-8 shadow-xl">
      <div className="text-center mb-12">
        <h2 className="text-3xl font-bold text-white mb-4">Upload & Detect Deepfakes</h2>
        <p className="text-gray-400">Upload an image or audio file to analyze whether it's AI-generated or real.</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Left Side: Upload */}
        <div className="space-y-6">
          <div className="border-2 border-dashed border-gray-700 rounded-xl p-8 text-center hover:border-[#6C5DD3] transition-colors group">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              onChange={handleFileUpload}
              accept="image/png,image/jpeg,image/jpg,audio/wav,audio/mp3"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer space-y-4 flex flex-col items-center"
            >
              <Upload className="h-12 w-12 text-[#6C5DD3] group-hover:scale-110 transition-transform" />
              <div className="space-y-2">
                <p className="text-white font-medium">Drag & drop or click to upload</p>
                <p className="text-sm text-gray-400">Supports PNG, JPG, WAV, MP3 files</p>
              </div>
            </label>
          </div>

          {file && (
            <div className="text-center">
              <p className="text-gray-300 mb-4">Selected file: {file.name}</p>
              <button
                onClick={handleDetect}
                disabled={isProcessing}
                className={`w-full py-3 rounded-full font-medium ${
                  isProcessing ? 'bg-gray-600 cursor-not-allowed' : 'bg-[#6C5DD3] hover:bg-[#8A7BF7]'
                } text-white transition-colors`}
              >
                {isProcessing ? 'Processing...' : 'Detect Fake'}
              </button>
            </div>
          )}
        </div>

        {/* Right Side: Results */}
        <div className="bg-[#232323] rounded-xl p-6">
          <h3 className="text-xl font-semibold text-white mb-6">Detection Results</h3>

          {error && (
            <div className="p-4 rounded-lg bg-red-900/20">
              <div className="flex items-center space-x-3">
                <AlertCircle className="h-6 w-6 text-red-400" />
                <span className="text-lg font-medium text-white">{error}</span>
              </div>
            </div>
          )}

          {result ? (
            <div className="space-y-6">
              <div className={`p-4 rounded-lg ${result.prediction === 'real' ? 'bg-green-900/20' : 'bg-red-900/20'}`}>
                <div className="flex items-center space-x-3">
                  <AlertCircle className={`h-6 w-6 ${result.prediction === 'real' ? 'text-green-400' : 'text-red-400'}`} />
                  <span className="text-lg font-medium text-white capitalize">{result.prediction}</span>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <p className="text-gray-400 mb-1">Confidence Score</p>
                  <div className="h-2 bg-gray-700 rounded-full">
                    <div
                      className={`h-full rounded-full ${result.prediction === 'real' ? 'bg-green-500' : 'bg-red-500'}`}
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                  <p className="text-right text-sm text-gray-400 mt-1">{result.confidence}%</p>
                </div>

                <div>
                  <p className="text-gray-400">File Path</p>
                  <p className="text-white break-all">{result.saved_file_path}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-400 py-12">
              <AlertCircle className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Upload a file to see detection results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UploadDetect;