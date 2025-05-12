
import React from 'react';
import { Brain, Twitter, Github, Linkedin } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <a href="/" className="flex items-center space-x-2">
              <Brain className="h-8 w-8 text-purple-500" />
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-purple-300">
                Detectify.ai
              </span>
            </a>
            <p className="text-gray-400">Protecting truth in the age of AI</p>
          </div>
          <div>
            <h3 className="text-white font-semibold mb-4">Product</h3>
            <ul className="space-y-2">
              <li><a href="#features" className="text-gray-400 hover:text-white">Features</a></li>
              <li><a href="#pricing" className="text-gray-400 hover:text-white">Pricing</a></li>
              <li><a href="#demo" className="text-gray-400 hover:text-white">Request Demo</a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-white font-semibold mb-4">Company</h3>
            <ul className="space-y-2">
              <li><a href="#careers" className="text-gray-400 hover:text-white">Careers</a></li>
              <li><a href="#contact" className="text-gray-400 hover:text-white">Contact</a></li>
            </ul>
          </div>
          <div>
            <h3 className="text-white font-semibold mb-4">Legal</h3>
            <ul className="space-y-2">
              <li><a href="#privacy" className="text-gray-400 hover:text-white">Privacy Policy</a></li>
              <li><a href="#terms" className="text-gray-400 hover:text-white">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-gray-800 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400">© 2025 Detectify.ai. All rights reserved.</p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#twitter" className="text-gray-400 hover:text-white transition-colors">
              <Twitter className="h-6 w-6" />
            </a>
            <a href="#github" className="text-gray-400 hover:text-white transition-colors">
              <Github className="h-6 w-6" />
            </a>
            <a href="#linkedin" className="text-gray-400 hover:text-white transition-colors">
              <Linkedin className="h-6 w-6" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
