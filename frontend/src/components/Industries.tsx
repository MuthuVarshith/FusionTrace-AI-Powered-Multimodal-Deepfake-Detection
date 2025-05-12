
import React from 'react';
import { FileCheck, Shield, UserCheck, AlertCircle } from 'lucide-react';

const Industries: React.FC = () => {
  return (
    <div className="mb-24 max-w-7xl mx-auto px-4 sm:px-6">
      <h2 className="text-3xl font-bold text-white text-center mb-16">Industries We Secure</h2>
      <div className="grid md:grid-cols-3 gap-8">
        <div className="bg-gray-800 p-8 rounded-2xl transform hover:scale-105 transition-transform duration-300">
          <FileCheck className="h-12 w-12 text-purple-500 mb-6" />
          <h3 className="text-xl font-bold text-white mb-4">Media & Content Authenticity</h3>
          <ul className="space-y-3 text-gray-400">
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Synthesized Voice Detection</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Content Integrity & Authenticity Verification</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Deepfake & AI-Generated Disinformation Protection</span>
            </li>
          </ul>
        </div>
        <div className="bg-gray-800 p-8 rounded-2xl transform hover:scale-105 transition-transform duration-300">
          <Shield className="h-12 w-12 text-purple-500 mb-6" />
          <h3 className="text-xl font-bold text-white mb-4">Finance & Cybersecurity</h3>
          <ul className="space-y-3 text-gray-400">
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>AI-Based Voice Cloning Detection</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Fraudulent Document & ID Verification</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Advanced KYC Protection</span>
            </li>
          </ul>
        </div>
        <div className="bg-gray-800 p-8 rounded-2xl transform hover:scale-105 transition-transform duration-300">
          <UserCheck className="h-12 w-12 text-purple-500 mb-6" />
          <h3 className="text-xl font-bold text-white mb-4">Government & Public Safety</h3>
          <ul className="space-y-3 text-gray-400">
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Voice Impersonation & Synthetic Speech Detection</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Detection of Fraudulent Communications</span>
            </li>
            <li className="flex items-start space-x-2">
              <AlertCircle className="h-5 w-5 text-purple-500 mt-1 flex-shrink-0" />
              <span>Combatting Disinformation Campaigns</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Industries;
