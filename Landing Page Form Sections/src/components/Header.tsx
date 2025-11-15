import { useState } from "react";
import { Button } from "./ui/button";
import { User, Bell, ArrowLeft, LogOut, RotateCcw } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface HeaderProps {
  onBackToLanding?: () => void;
  onLogout?: () => void;
  onClearForm?: () => void;
}

export function Header({ onBackToLanding, onLogout, onClearForm }: HeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const userName = localStorage.getItem("userName") || "User";
  const userEmail = localStorage.getItem("userEmail") || "";

  useState(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  });

  return (
    <header className={`border-b border-indigo-200 bg-white/95 backdrop-blur-md px-8 py-4 flex items-center justify-between sticky top-0 z-50 transition-all duration-300 ${
      scrolled ? "shadow-lg" : "shadow-sm"
    }`}>
      <div className="flex items-center gap-3">
        {onBackToLanding && (
          <motion.button
            onClick={onBackToLanding}
            whileHover={{ scale: 1.1, x: -2 }}
            whileTap={{ scale: 0.95 }}
            className="p-2 hover:bg-indigo-50 rounded-xl transition-all duration-200 mr-1"
            title="Back to landing page"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </motion.button>
        )}
        <motion.div 
          whileHover={{ scale: 1.05, rotate: 5 }}
          className="w-10 h-10 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg"
        >
          <span className="text-white font-bold text-base">BG</span>
        </motion.div>
        <div className="flex flex-col">
          <span className="font-bold text-gray-800 text-lg">BrandGen</span>
          <span className="text-xs text-gray-500">AI-Powered Branding</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {onClearForm && (
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              onClick={onClearForm}
              variant="outline"
              className="border-2 border-gray-300 hover:border-red-400 hover:bg-red-50 text-gray-700 hover:text-red-600 transition-all duration-300 rounded-xl shadow-sm hover:shadow-md"
            >
              <span className="flex items-center gap-2">
                <RotateCcw className="w-4 h-4" />
                Clear Form
              </span>
            </Button>
          </motion.div>
        )}
        
        <motion.button 
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="p-2.5 hover:bg-indigo-50 rounded-xl transition-all duration-200 relative group"
        >
          <Bell className="w-5 h-5 text-gray-600 group-hover:text-indigo-600 transition-colors" />
          <motion.span 
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="absolute top-1.5 right-1.5 w-2.5 h-2.5 bg-red-500 rounded-full shadow-lg"
          />
        </motion.button>

        <div className="relative">
          <motion.button
            onClick={() => setShowUserMenu(!showUserMenu)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center text-white shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer ring-2 ring-white"
          >
            <User className="w-5 h-5" />
          </motion.button>

          <AnimatePresence>
            {showUserMenu && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.95 }}
                transition={{ duration: 0.2, type: "spring" }}
                className="absolute right-0 mt-3 w-72 bg-white rounded-2xl shadow-2xl border border-indigo-100 overflow-hidden"
              >
                <div className="p-5 border-b border-indigo-100 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-md">
                      <User className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="font-bold text-gray-800">{userName}</p>
                      <p className="text-xs text-gray-600">{userEmail}</p>
                    </div>
                  </div>
                </div>
                {onLogout && (
                  <motion.button
                    onClick={() => {
                      setShowUserMenu(false);
                      onLogout();
                    }}
                    whileHover={{ backgroundColor: "rgb(238 242 255)" }}
                    className="w-full p-4 flex items-center gap-3 transition-colors text-left text-gray-700"
                  >
                    <LogOut className="w-5 h-5 text-gray-600" />
                    <span className="font-medium">Sign Out</span>
                  </motion.button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}
