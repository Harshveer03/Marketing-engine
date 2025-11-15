import { motion } from "motion/react";
import { Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-gradient-to-r from-white via-indigo-50/30 to-white px-8 py-4 flex items-center justify-between shrink-0 border-t border-indigo-100">
      <div className="flex items-center gap-2">
        <p className="text-gray-600 text-sm m-0 flex items-center gap-2">
          © 2024 BrandGen. Made with 
          <motion.span
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <Heart className="w-4 h-4 text-red-500 fill-red-500" />
          </motion.span>
          for amazing brands
        </p>
      </div>

      <div className="flex items-center gap-8">
        <motion.a
          href="#"
          whileHover={{ y: -2 }}
          className="text-gray-600 hover:text-indigo-600 transition-all duration-200 text-sm font-medium relative group"
        >
          Privacy Policy
          <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 group-hover:w-full transition-all duration-300"></span>
        </motion.a>
        <motion.a
          href="#"
          whileHover={{ y: -2 }}
          className="text-gray-600 hover:text-indigo-600 transition-all duration-200 text-sm font-medium relative group"
        >
          Terms of Service
          <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 group-hover:w-full transition-all duration-300"></span>
        </motion.a>
        <motion.a
          href="#"
          whileHover={{ y: -2 }}
          className="text-gray-600 hover:text-indigo-600 transition-all duration-200 text-sm font-medium relative group"
        >
          Contact Us
          <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-500 to-purple-500 group-hover:w-full transition-all duration-300"></span>
        </motion.a>
      </div>
    </footer>
  );
}
