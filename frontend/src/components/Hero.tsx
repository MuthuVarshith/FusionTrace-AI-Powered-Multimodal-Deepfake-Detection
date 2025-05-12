
import React, { useEffect, useRef } from 'react';

const Hero: React.FC = () => {
  const typedRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const strings = ['Fraud_', 'Generative AI_', 'Disinformation_', 'Voice Spoofing_'];
    let currentStringIndex = 0;
    let currentCharIndex = 0;
    let isDeleting = false;

    const type = () => {
      if (typedRef.current) {
        const currentString = strings[currentStringIndex];
        if (isDeleting) {
          typedRef.current.textContent = currentString.substring(0, currentCharIndex - 1);
          currentCharIndex--;
          if (currentCharIndex === 0) {
            isDeleting = false;
            currentStringIndex = (currentStringIndex + 1) % strings.length;
          }
        } else {
          typedRef.current.textContent = currentString.substring(0, currentCharIndex + 1);
          currentCharIndex++;
          if (currentCharIndex === currentString.length) {
            isDeleting = true;
            setTimeout(type, 1000);
            return;
          }
        }
        setTimeout(type, isDeleting ? 50 : 100);
      }
    };

    type();
  }, []);

  return (
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 z-0" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 pt-32 pb-16 relative z-10">
        <div className="text-center">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold mb-4">
            <span className="text-white">Detect</span>
          </h1>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-8">
            <span ref={typedRef} className="bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-purple-300"></span>
          </h2>
          <p className="max-w-2xl mx-auto text-xl mt-8 text-gray-300 mb-10">
            Detectify.ai's multi-model and multimodal deepfake detection platform helps enterprises
            and governments detect AI-generated content at scale.
          </p>
          <div className="flex gap-8 mt-12 items-center justify-center">
            <a href="#detect" className="bg-purple-500 text-white px-8 py-3 rounded-lg hover:bg-purple-400 transition-colors">
              Try it Now
            </a>
            <a href="#how-it-works" className="text-gray-300 hover:text-white">
              Learn More →
            </a>
          </div>
        </div>
        <div className="mt-16 mb-24">
          <div className="overflow-x-auto flex gap-4 snap-x snap-mandatory">
            {[
              { src: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=800&q=80', label: 'Real Image', status: 'Verified Authentic', color: 'text-green-400' },
              { src: 'https://images.unsplash.com/photo-1552058544-f2b08422138a?auto=format&fit=crop&w=800&q=80', label: 'Real Image', status: 'Verified Authentic', color: 'text-green-400' },
              { src: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?auto=format&fit=crop&w=800&q=80', label: 'AI Generated', status: 'Detected as Fake', color: 'text-red-400' },
              { src: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=800&q=80', label: 'AI Generated', status: 'Detected as Fake', color: 'text-red-400' },
            ].map((item, index) => (
              <div key={index} className="snap-center flex-shrink-0 w-64">
                <div className="relative group">
                  <img src={item.src} alt={item.label} className="rounded-lg h-64 w-full object-cover transition-transform duration-300 group-hover:scale-105" />
                  <p className="text-center text-white mt-4 text-lg font-semibold">{item.label}</p>
                  <p className={`text-center text-sm ${item.color}`}>{item.status}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
