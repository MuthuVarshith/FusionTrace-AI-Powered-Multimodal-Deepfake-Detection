
import React from 'react';
import { Mail, Phone, MapPin } from 'lucide-react';

const Contact: React.FC = () => {
  return (
    <div id="contact" className="mb-8 bg-gray-800 rounded-2xl p-8 shadow-xl max-w-7xl mx-auto px-4 sm:px-6">
      <h2 className="text-3xl font-bold text-white text-center mb-12">Contact Us</h2>
      <div className="flex flex-row justify-between space-x-12 px-4 md:px-20 pb-8">
        <div className="flex space-x-4">
          <Mail className="h-6 w-6 text-purple-500 mt-1" />
          <div>
            <h3 className="text-white font-semibold mb-2">Email</h3>
            <p className="text-gray-400">support@detectify.ai</p>
            <p className="text-gray-400">business@detectify.ai (Business)</p>
            <p className="text-gray-400">press@detectify.ai (Media)</p>
          </div>
        </div>
        <div className="flex items-start space-x-4">
          <Phone className="h-6 w-6 text-purple-500 mt-1" />
          <div>
            <h3 className="text-white font-semibold mb-2">Phone</h3>
            <p className="text-gray-400">+91-9999-999999</p>
          </div>
        </div>
        <div className="flex items-start space-x-4">
          <MapPin className="h-6 w-6 text-purple-500 mt-1" />
          <div>
            <h3 className="text-white font-semibold mb-2">Address</h3>
            <p className="text-gray-400">Pune, India</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;
