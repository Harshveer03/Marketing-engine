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
    <div className="max-w-6xl mx-auto h-full flex flex-col">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-10 py-4 px-12"
      >
        <motion.h2
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl mb-3 text-white font-bold"
        >
          Your Brand Assessment
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-white text-xl"
        >
          Here's how your brand is shaping up
        </motion.p>
      </motion.div>

      <div className="max-w-3xl mx-auto mb-3">
        {/* Score Display */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white p-8 shadow-2xl hover:shadow-3xl transition-shadow duration-300 border-2 border-black rounded-3xl"
        >
          <div className="flex items-center gap-4 mb-6 pb-4 border-b border-white">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="bg-black w-14 h-14 rounded-2xl flex items-center justify-center shadow-xl"
            >
              <TrendingUp className="w-7 h-7 text-black" />
            </motion.div>
            <h3 className="text-3xl text-gray-800 font-bold">Your Score</h3>
          </div>

          {/* Main Score - Square */}
          <div className="text-center py-3 mb-3 flex flex-col items-center justify-center">
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
              <div className="w-32 h-32 bg-black flex items-center justify-center shadow-xl rounded-2xl">
                <div className="w-28 h-28 bg-white rounded-xl flex flex-col items-center justify-center">
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.8 }}
                    className="text-5xl font-bold text-gray-900"
                  >
                    85
                  </motion.span>
                  <span className="text-gray-500 text-sm font-medium">
                    out of 100
                  </span>
                </div>
              </div>
              
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
              className="text-gray-700 mt-4 font-medium text-base"
            >
              Great progress! You're on the right track.
            </motion.p>
          </div>

          {/* Score Breakdown */}
          <div className="space-y-2">
            <h4 className="text-lg font-bold text-gray-700 mb-2">
              Score Breakdown
            </h4>
            {scoreBreakdown.map((item, index) => (
              <motion.div
                key={item.label}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 1 + index * 0.1 }}
                className="bg-gray-50 p-5 rounded-xl"
              >
                <div className="flex justify-between items-center mb-2">
                  <span className="text-base text-gray-900 font-semibold">
                    {item.label}
                  </span>
                  <span className="text-base font-bold text-black">
                    {item.score}%
                  </span>
                </div>
                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${item.score}%` }}
                    transition={{ duration: 0.8, delay: 1.2 + index * 0.1 }}
                    className="h-full bg-black rounded-full"
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
        className="flex justify-between items-center bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200"
      >
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-1">
            Ready to generate your brand?
          </h4>
          <p className="text-gray-600 text-sm">
            Complete all sections to unlock full potential
          </p>
        </div>
        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button className="!bg-black hover:!bg-gray-800 text-white shadow-lg hover:shadow-xl border-0 px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-200">
            <span className="flex items-center gap-3">
              <Sparkles className="w-5 h-5" />
              Let's Generate Your Brand
            </span>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
