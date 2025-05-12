import React, { useState } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import MultimodalDemo from './components/MultimodalDemo';
import UploadDetect from './components/UploadDetect';
import HowItWorks from './components/HowItWorks';
import Industries from './components/Industries';
import Contact from './components/Contact';
import Footer from './components/Footer';
import AuthModal from './components/AuthModal';

const App: React.FC = () => {
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);

  return (
    <div className="min-h-screen bg-[#0D0D0D] text-white">
      {showLogin && (
        <AuthModal
          isLogin={true}
          onClose={() => setShowLogin(false)}
          setShowLogin={setShowLogin}
          setShowRegister={setShowRegister}
        />
      )}
      {showRegister && (
        <AuthModal
          isLogin={false}
          onClose={() => setShowRegister(false)}
          setShowLogin={setShowLogin}
          setShowRegister={setShowRegister}
        />
      )}
      
      <Header setShowLogin={setShowLogin} setShowRegister={setShowRegister} />
      <main>
        <Hero />
        <MultimodalDemo />
        <UploadDetect />
        <HowItWorks />
        <Industries />
        <Contact />
      </main>
      <Footer />
    </div>
  );
};

export default App;