
import React, { useState } from 'react';
import { Brain, Menu, X } from 'lucide-react';

interface HeaderProps {
  setShowLogin: (show: boolean) => void;
  setShowRegister: (show: boolean) => void;
}

const Header: React.FC<HeaderProps> = ({ setShowLogin, setShowRegister }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="fixed w-full bg-gray-900/80 backdrop-blur-md z-50 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center space-x-8">
            <a href="/" className="flex items-center space-x-2">
              <Brain className="h-8 w-8 text-purple-500" />
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-purple-300">Detectify.ai</span>
            </a>
            <nav className="hidden md:flex items-center space-x-8">
              <a href="#how-it-works" className="text-gray-300 hover:text-white">How It Works</a>
              <a href="#detect" className="text-gray-300 hover:text-white">Detect</a>
              <a href="#contact" className="text-gray-300 hover:text-white">Contact Us</a>
            </nav>
          </div>
          <div className="hidden md:flex items-center space-x-6">
            <button onClick={() => setShowLogin(true)} className="text-gray-300 hover:text-white">
              Log in
            </button>
            <button onClick={() => setShowRegister(true)} className="bg-purple-500 hover:bg-purple-400 text-white px-6 py-2 rounded-md transition-colors">
              Get Started Now
            </button>
          </div>
          <button className="md:hidden" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            {isMenuOpen ? <X className="h-6 w-6 text-gray-300" /> : <Menu className="h-6 w-6 text-gray-300" />}
          </button>
        </div>
      </div>
      {isMenuOpen && (
        <div className="md:hidden bg-gray-900">
          <div className="px-2 pt-2 pb-3 space-y-1">
            <a href="#how-it-works" className="block px-3 py-2 text-gray-300 hover:text-white">How It Works</a>
            <a href="#detect" className="block px-3 py-2 text-gray-300 hover:text-white">Detect</a>
            <a href="#contact" className="block px-3 py-2 text-gray-300 hover:text-white">Contact Us</a>
            <button onClick={() => setShowLogin(true)} className="block w-full text-left px-3 py-2 text-gray-300 hover:text-white">
              Log in
            </button>
            <button onClick={() => setShowRegister(true)} className="block w-full text-left px-3 py-2 text-purple-500 hover:text-purple-400">
              Get Started Now
            </button>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
