import { useState } from "react";
import { Button } from "./ui/button";
import { ArrowRight, Sparkles, RefreshCw, Rocket, Target } from "lucide-react";
import { motion } from "motion/react";

const options = [
  {
    id: "refine",
    label: "Refine",
    icon: Sparkles,
    description: "Polish and enhance your existing brand",
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: "redefine",
    label: "Redefine",
    icon: RefreshCw,
    description: "Transform your brand identity",
    color: "from-purple-500 to-pink-500",
  },
  {
    id: "from-scratch",
    label: "From Scratch",
    icon: Rocket,
    description: "Build a brand from the ground up",
    color: "from-indigo-500 to-purple-500",
  },
  {
    id: "general-use",
    label: "General Use",
    icon: Target,
    description: "Explore brand possibilities",
    color: "from-emerald-500 to-teal-500",
  },
];

interface WhatUseSectionProps {
  onNext: () => void;
  selectedOption: string | null;
  onSelectOption: (option: string) => void;
}

export function WhatUseSection({
  onNext,
  selectedOption,
  onSelectOption,
}: WhatUseSectionProps) {
  return (
    <div className="max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-16 py-8 px-12"
      >
        <h2 className="text-5xl mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          What do you want to use us for?
        </h2>
        <p className="text-gray-600 text-xl">
          Choose the option that best fits your needs
        </p>
      </motion.div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        {options.map((option, index) => {
          return (
            <motion.button
              key={option.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              whileHover={{ y: -8, scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onSelectOption(option.id)}
              className={`group relative p-12 transition-all duration-300 flex flex-col items-center justify-center text-center ${
                selectedOption === option.id
                  ? "bg-white border-2 border-indigo-500 shadow-2xl"
                  : "bg-white border-2 border-gray-200 hover:border-indigo-300 shadow-lg hover:shadow-xl"
              }`}
            >
              {/* Label */}
              <h3
                className={`text-2xl mb-2 transition-colors ${
                  selectedOption === option.id
                    ? "text-indigo-700"
                    : "text-gray-800"
                }`}
              >
                {option.label}
              </h3>

              {/* Description */}
              <p className="text-gray-600 text-sm">{option.description}</p>

              {/* Selected indicator */}
              {selectedOption === option.id && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="absolute top-4 right-4 w-8 h-8 bg-indigo-500 flex items-center justify-center"
                >
                  <svg
                    className="w-5 h-5 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={3}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </motion.div>
              )}
            </motion.button>
          );
        })}
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="flex justify-between items-center"
      >
        <p className="text-gray-500 text-sm">
          {selectedOption
            ? "✓ Selection made"
            : "Please select an option to continue"}
        </p>
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button
            onClick={onNext}
            disabled={!selectedOption}
            className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg border-0 px-8 py-6 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="flex items-center gap-2">
              Continue <ArrowRight className="w-5 h-5" />
            </span>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
