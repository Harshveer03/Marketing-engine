import { useState } from "react";
import { Button } from "./ui/button";
import { User, Bell, ArrowLeft, LogOut } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface HeaderProps {
  onBackToLanding?: () => void;
  onLogout?: () => void;
}

export function Header({ onBackToLanding, onLogout }: HeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const userName = localStorage.getItem("userName") || "User";
  const userEmail = localStorage.getItem("userEmail") || "";

  return (
    <header className="border-b border-indigo-200 bg-white/95 backdrop-blur-md px-8 py-3 flex items-center justify-between shadow-sm sticky top-0 z-50">
      <div className="flex items-center gap-2">
        {onBackToLanding && (
          <button
            onClick={onBackToLanding}
            className="p-2 hover:bg-indigo-50 rounded-lg transition-all duration-200 mr-2"
            title="Back to landing page"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
        )}
        <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
          <span className="text-white font-bold text-sm">BG</span>
        </div>
        <span className="font-semibold text-gray-800">BrandGen</span>
      </div>

      <div className="flex items-center gap-3">
        <button className="p-2 hover:bg-indigo-50 rounded-lg transition-all duration-200 relative">
          <Bell className="w-5 h-5 text-gray-600" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
        </button>

        <div className="relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-md hover:scale-110 transition-transform duration-200 cursor-pointer"
          >
            <User className="w-5 h-5" />
          </button>

          <AnimatePresence>
            {showUserMenu && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
                className="absolute right-0 mt-2 w-64 bg-white rounded-xl shadow-2xl border border-indigo-100 overflow-hidden"
              >
                <div className="p-4 border-b border-indigo-100 bg-gradient-to-br from-indigo-50 to-purple-50">
                  <p className="font-semibold text-gray-800">{userName}</p>
                  <p className="text-sm text-gray-600">{userEmail}</p>
                </div>
                {onLogout && (
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      onLogout();
                    }}
                    className="w-full p-4 flex items-center gap-3 hover:bg-indigo-50 transition-colors text-left text-gray-700"
                  >
                    <LogOut className="w-5 h-5 text-gray-600" />
                    <span>Sign Out</span>
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
}
