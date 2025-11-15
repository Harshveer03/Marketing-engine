import {
  CheckCircle2,
  Circle,
  Sparkles,
  TrendingUp,
  Award,
  Target,
} from "lucide-react";
import { Button } from "./ui/button";
import { motion } from "motion/react";

export function ScoreSection() {
  const sections = [
    { name: "What Use?", completed: true },
    { name: "Brand Info & Resources", completed: true },
    { name: "Score", completed: false },
  ];

  const scoreBreakdown = [
    { label: "Brand Identity", score: 90, color: "from-blue-500 to-cyan-500" },
    {
      label: "Target Audience",
      score: 85,
      color: "from-purple-500 to-pink-500",
    },
    {
      label: "Resources Quality",
      score: 80,
      color: "from-indigo-500 to-purple-500",
    },
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-6 py-0 px-12"
      >
        <h2 className="text-5xl mb-0.5 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent font-bold">
          Your Brand Assessment
        </h2>
        <p className="text-gray-600 text-xl">
          Here's how your brand is shaping up
        </p>
      </motion.div>

      <div className="grid grid-cols-2 gap-4 mb-3">
        {/* Left Column - Completion Status */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white p-6 shadow-lg border border-gray-100"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
              <CheckCircle2 className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-2xl text-gray-800 font-semibold">
              Completion Status
            </h3>
          </div>

          <div className="space-y-3">
            {sections.map((section, index) => (
              <motion.div
                key={section.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
                className={`flex items-center gap-3 p-3 transition-all duration-200 ${
                  section.completed
                    ? "bg-green-50 border-2 border-green-200"
                    : "bg-gray-50 border-2 border-gray-200"
                }`}
              >
                {section.completed ? (
                  <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{
                      type: "spring",
                      stiffness: 200,
                      delay: 0.5 + index * 0.1,
                    }}
                    className="flex-shrink-0"
                  >
                    <CheckCircle2 className="w-6 h-6 text-green-600" />
                  </motion.div>
                ) : (
                  <Circle className="w-6 h-6 text-gray-400 flex-shrink-0" />
                )}
                <span
                  className={`font-medium text-base ${
                    section.completed ? "text-green-700" : "text-gray-600"
                  }`}
                >
                  {section.name}
                </span>
              </motion.div>
            ))}
          </div>

          {/* Progress Bar */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex justify-between items-center mb-2">
              <span className="text-base text-gray-600 font-medium">
                Overall Progress
              </span>
              <span className="text-base font-semibold text-indigo-600">
                67%
              </span>
            </div>
            <div className="h-4 bg-gray-200 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "67%" }}
                transition={{ duration: 1, delay: 0.5 }}
                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
              />
            </div>
          </div>
        </motion.div>

        {/* Right Column - Score Display */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-6 shadow-lg border border-indigo-100"
        >
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-2xl text-gray-800 font-semibold">Your Score</h3>
          </div>

          {/* Main Score - Square */}
          <div className="text-center py-4 mb-4 flex flex-col items-center justify-center">
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{
                type: "spring",
                stiffness: 200,
                damping: 15,
                delay: 0.5,
              }}
              className="relative inline-block"
            >
              <div className="w-32 h-32 bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-2xl">
                <div className="w-28 h-28 bg-white flex flex-col items-center justify-center">
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="text-5xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent"
                  >
                    85
                  </motion.span>
                  <span className="text-gray-500 text-sm font-medium">
                    out of 100
                  </span>
                </div>
              </div>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 1, type: "spring" }}
                className="absolute -top-1 -right-1 w-8 h-8 bg-yellow-400 flex items-center justify-center shadow-lg"
              >
                <Sparkles className="w-4 h-4 text-white" />
              </motion.div>
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
              className="text-gray-700 mt-3 font-medium text-base"
            >
              Great progress! You're on the right track.
            </motion.p>
          </div>

          {/* Score Breakdown */}
          <div className="space-y-3">
            <h4 className="text-base font-semibold text-gray-700 mb-2">
              Score Breakdown
            </h4>
            {scoreBreakdown.map((item, index) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1 + index * 0.1 }}
                className="bg-white p-3 shadow-sm"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-base text-gray-700 font-medium">
                    {item.label}
                  </span>
                  <span className="text-base font-bold text-indigo-600">
                    {item.score}%
                  </span>
                </div>
                <div className="h-3 bg-gray-200 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${item.score}%` }}
                    transition={{ duration: 0.8, delay: 1.2 + index * 0.1 }}
                    className={`h-full bg-gradient-to-r ${item.color}`}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5 }}
        className="flex justify-between items-center bg-white p-5 shadow-lg border border-gray-100"
      >
        <div>
          <h4 className="text-base font-semibold text-gray-800 mb-0.5">
            Ready to generate your brand?
          </h4>
          <p className="text-gray-600 text-xs">
            Complete all sections to unlock full potential
          </p>
        </div>
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white shadow-lg border-0 px-6 py-3">
            <span className="flex items-center gap-2">
              <Sparkles className="w-3 h-4" />
              Let's Generate
            </span>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
