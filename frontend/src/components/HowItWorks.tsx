
import React from 'react';
import { Upload, Brain, AlertCircle, Zap, Shield, Lock } from 'lucide-react';

const HowItWorks: React.FC = () => {
  return (
    <div id="how-it-works" className="mb-24 max-w-7xl mx-auto px-4 sm:px-6">
      <h2 className="text-3xl font-bold text-white text-center mb-16">How It Works</h2>
      <div className="relative">
        <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-1 bg-purple-500 opacity-20"></div>
        <div className="space-y-24">
          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-1/2 -mt-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">1</span>
              </div>
            </div>
            <div className="ml-auto mr-[calc(50%+2rem)] md:mr-[calc(50%+4rem)] p-6 bg-gray-800 rounded-xl">
              <h3 className="text-xl font-bold text-white mb-4">Upload Media</h3>
              <ul className="space-y-2 text-gray-400">
                <li className="flex items-start space-x-2">
                  <Upload className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Upload an image or audio file for analysis</span>
                </li>
                <li className="flex items-start space-x-2">
                  <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Supports WAV, MP3, PNG, JPG, and JPEG formats</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-1/2 -mt-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">2</span>
              </div>
            </div>
            <div className="mr-auto ml-[calc(50%+2rem)] md:ml-[calc(50%+4rem)] p-6 bg-gray-800 rounded-xl">
              <h3 className="text-xl font-bold text-white mb-4">AI Detection in Action</h3>
              <ul className="space-y-2 text-gray-400">
                <li className="flex items-start space-x-2">
                  <Brain className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Our advanced AI scans the media using deep learning models</span>
                </li>
                <li className="flex items-start space-x-2">
                  <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Analyzes patterns, inconsistencies, and digital manipulations</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-1/2 -mt-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">3</span>
              </div>
            </div>
            <div className="ml-auto mr-[calc(50%+2rem)] md:mr-[calc(50%+4rem)] p-6 bg-gray-800 rounded-xl">
              <h3 className="text-xl font-bold text-white mb-4">Get Instant Results</h3>
              <ul className="space-y-2 text-gray-400">
                <li className="flex items-start space-x-2">
                  <Zap className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>The system provides a confidence score indicating if the content is real or fake</span>
                </li>
                <li className="flex items-start space-x-2">
                  <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Visual reports highlight detected anomalies</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-1/2 -mt-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">4</span>
              </div>
            </div>
            <div className="mr-auto ml-[calc(50%+2rem)] md:ml-[calc(50%+4rem)] p-6 bg-gray-800 rounded-xl">
              <h3 className="text-xl font-bold text-white mb-4">Secure & Reliable</h3>
              <ul className="space-y-2 text-gray-400">
                <li className="flex items-start space-x-2">
                  <Shield className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Uses cutting-edge AI models for high accuracy</span>
                </li>
                <li className="flex items-start space-x-2">
                  <Lock className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
                  <span>Ensures data privacy and secure processing</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HowItWorks;
