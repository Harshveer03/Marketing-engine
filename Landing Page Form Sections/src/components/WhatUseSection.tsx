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
    <div className="max-w-6xl mx-auto h-full flex flex-col justify-between">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12 py-6 px-12"
      >
        <motion.h2
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-bold mb-4 text-gray-900"
        >
          What do you want to use us for?
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-gray-600 text-xl"
        >
          Choose the option that best fits your needs
        </motion.p>
      </motion.div>

      <div className="grid grid-cols-2 gap-8 flex-1">
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
              className={`group relative p-16 transition-all duration-300 flex flex-col items-center justify-center text-center rounded-2xl ${
                selectedOption === option.id
                  ? "bg-black border-4 border-black shadow-2xl"
                  : "bg-white border-4 border-gray-300 hover:border-gray-400 shadow-lg hover:shadow-xl"
              }`}
            >
              {/* Label */}
              <h3
                className={`text-3xl font-bold mb-3 transition-colors ${
                  selectedOption === option.id ? "text-white" : "text-gray-900"
                }`}
              >
                {option.label}
              </h3>

              {/* Description */}
              <p
                className={`text-base ${
                  selectedOption === option.id
                    ? "text-gray-300"
                    : "text-gray-600"
                }`}
              >
                {option.description}
              </p>

              {/* Selected indicator */}
              {selectedOption === option.id && (
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ type: "spring", stiffness: 200 }}
                  className="absolute top-6 right-6 w-12 h-12 bg-white rounded-xl flex items-center justify-center shadow-lg"
                >
                  <svg
                    className="w-7 h-7 text-black"
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
        className="flex justify-between items-center mt-8"
      >
        <p className="text-gray-500 text-sm">
          {selectedOption
            ? "✓ Selection made"
            : "Please select an option to continue"}
        </p>
        <motion.div
          whileHover={{
            scale: selectedOption ? 1.05 : 1,
            x: selectedOption ? 5 : 0,
          }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            onClick={onNext}
            disabled={!selectedOption}
            className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 text-white shadow-xl hover:shadow-2xl border-0 px-10 py-6 rounded-xl text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            <span className="flex items-center gap-3">
              Continue to Next Step
              {selectedOption && (
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                >
                  <ArrowRight className="w-5 h-5" />
                </motion.div>
              )}
            </span>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
