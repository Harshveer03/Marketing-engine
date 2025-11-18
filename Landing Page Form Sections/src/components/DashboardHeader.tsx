import { useState } from "react";
import { Button } from "./ui/button";
import { User, ArrowLeft, CreditCard } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface DashboardHeaderProps {
  onBackToLanding?: () => void;
  creditsLeft?: number;
}

export function DashboardHeader({
  onBackToLanding,
  creditsLeft = 25,
}: DashboardHeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const userName = localStorage.getItem("userName") || "User";
  const userEmail = localStorage.getItem("userEmail") || "";

  return (
    <header className="border-b border-gray-200 bg-white/95 backdrop-blur-md px-6 py-3 flex items-center justify-between sticky top-0 z-50 shadow-sm">
      {/* Left Side */}
      <div className="flex items-center gap-2.5">
        {onBackToLanding && (
          <motion.button
            onClick={onBackToLanding}
            whileHover={{ scale: 1.1, x: -2 }}
            whileTap={{ scale: 0.95 }}
            className="p-1.5 hover:bg-gray-100 rounded-xl transition-all duration-200 mr-0.5"
            title="Back to landing page"
          >
            <ArrowLeft className="w-4 h-4 text-gray-600" />
          </motion.button>
        )}
        <motion.div whileHover={{ scale: 1.05 }}>
          <img src="/1syx-logo.jpeg" alt="1SYX Logo" className="h-8 w-auto" />
        </motion.div>
        <div className="flex flex-col">
          <span className="font-bold text-gray-900 text-base tracking-wider">
            1SYX
          </span>
          <span className="text-[10px] text-gray-600 uppercase tracking-wide">
            1-System For Your 'X' Factor
          </span>
        </div>
      </div>

      {/* Right Side */}
      <div className="flex items-center gap-3">
        {/* Credits Left Card */}
        <motion.div
          whileHover={{ scale: 1.02 }}
          className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-xl px-3 py-2"
        >
          <CreditCard className="w-4 h-4 text-gray-600" />
          <div className="flex flex-col">
            <span className="text-[10px] text-gray-500 uppercase">Credits</span>
            <span className="text-sm font-bold text-gray-900">
              {creditsLeft}
            </span>
          </div>
        </motion.div>

        {/* Upgrade Button */}
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button className="!bg-black hover:bg-gray-800 text-white border-0 shadow-md hover:shadow-lg rounded-xl text-sm py-2 px-4 font-semibold">
            Upgrade
          </Button>
        </motion.div>

        {/* Profile Button */}
        <div className="relative">
          <motion.button
            onClick={() => setShowUserMenu(!showUserMenu)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            className="w-8 h-8 rounded-full bg-black flex items-center justify-center text-black shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer ring-2 ring-white"
          >
            <User className="w-4 h-4" />
          </motion.button>

          <AnimatePresence>
            {showUserMenu && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.2, type: "spring" }}
                className="absolute right-0 mt-3 w-72 bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden"
              >
                <div className="p-5 border-b border-gray-200 bg-gray-50">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-12 h-12 rounded-full bg-black flex items-center justify-center text-black shadow-md">
                      <User className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="font-bold text-gray-800">{userName}</p>
                      <p className="text-xs text-gray-600">{userEmail}</p>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  <p className="text-sm text-gray-600">
                    Profile settings coming soon...
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}
