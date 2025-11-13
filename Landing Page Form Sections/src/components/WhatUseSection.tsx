import { useState } from 'react';
import { Button } from './ui/button';
import { ArrowRight } from 'lucide-react';
import { motion } from 'motion/react';

const options = [
  { id: 'refine', label: 'Refine' },
  { id: 'redefine', label: 'Redefine' },
  { id: 'from-scratch', label: 'From Scratch' },
  { id: 'general-use', label: 'General Use' },
];

interface WhatUseSectionProps {
  onNext: () => void;
}

export function WhatUseSection({ onNext }: WhatUseSectionProps) {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  return (
    <div className="max-w-4xl">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-2xl p-12 mb-8 shadow-xl relative overflow-hidden"
      >
        <div 
          className="absolute inset-0 opacity-20 bg-cover bg-center"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
          }}
        />
        <h2 className="text-center relative z-10">What do you want to use us for?</h2>
      </motion.div>
      
      <div className="grid grid-cols-2 gap-8 mb-8">
        {options.map((option, index) => (
          <motion.button
            key={option.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setSelectedOption(option.id)}
            className={`border-2 rounded-2xl p-12 transition-all duration-300 shadow-lg relative overflow-hidden ${
              selectedOption === option.id
                ? 'bg-gradient-to-br from-indigo-100 to-purple-100 border-indigo-400 shadow-xl'
                : 'bg-white border-indigo-200 hover:border-indigo-300 hover:shadow-xl'
            }`}
          >
            {selectedOption === option.id && (
              <div 
                className="absolute inset-0 opacity-10 bg-cover bg-center"
                style={{
                  backgroundImage: `url('https://images.unsplash.com/photo-1675636173835-657f72144c77?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzb2Z0JTIwdGV4dHVyZSUyMGJhY2tncm91bmR8ZW58MXx8fHwxNzYzMDIyMjUwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                }}
              />
            )}
            <span className="relative z-10">{option.label}</span>
          </motion.button>
        ))}
      </div>
      
      <div className="flex justify-end">
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button 
            onClick={onNext}
            className="bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 text-white rounded-xl shadow-lg border-0 relative overflow-hidden"
          >
            <div 
              className="absolute inset-0 opacity-30 bg-cover bg-center"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
              }}
            />
            <span className="relative z-10 flex items-center">
              Next <ArrowRight className="ml-2 w-4 h-4" />
            </span>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}