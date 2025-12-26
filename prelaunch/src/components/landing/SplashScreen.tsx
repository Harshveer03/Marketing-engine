import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";

interface SplashScreenProps {
  onComplete: () => void;
}

export function SplashScreen({ onComplete }: SplashScreenProps) {
  const [isAnimating, setIsAnimating] = useState(false);



  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50"
        initial={{ opacity: 1 }}
        animate={{ opacity: isAnimating ? 0 : 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.5, delay: isAnimating ? 1.2 : 0 }}
        style={{
          backgroundColor: "#000000",
          width: "100vw",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >


        {/* Click Prompt - Top Center */}
        <motion.div
          className="absolute left-0 right-0 text-center z-10"
          style={{ top: "80px" }}
          animate={isAnimating ? { opacity: 0 } : { opacity: [1, 1, 1] }}
          transition={
            isAnimating
              ? { duration: 0.3 }
              : { duration: 2, repeat: Infinity, ease: "easeInOut" }
          }
        >
          <h1
            className="text-white text-5xl font-bold tracking-widest"
            style={{ textShadow: "0 0 20px rgba(255,255,255,0.5)" }}
          >
            Let Us Bring You To The LIGHT
          </h1>
        </motion.div>

        {/* Full Logo Image - Center */}
        <motion.div
          className="relative"
          animate={isAnimating ? { rotateY: -90, x: -200 } : {}}
          transition={{ duration: 1.2, ease: "easeInOut" }}
          style={{
            transformOrigin: "left center",
            transformStyle: "preserve-3d",
          }}
        >
          <video
            src="/1syx-logo-animation.mp4"
            autoPlay
            muted
            playsInline
            onEnded={() => setIsAnimating(true)}
            className="max-w-2xl w-full h-auto object-contain"
          />
        </motion.div>

        {/* Zoom Effect */}
        {isAnimating && (
          <motion.div
            className="fixed inset-0 pointer-events-none"
            initial={{ scale: 1 }}
            animate={{ scale: 3 }}
            transition={{ duration: 1.2, delay: 0.6, ease: "easeInOut" }}
            onAnimationComplete={onComplete}
          />
        )}
      </motion.div>
    </AnimatePresence>
  );
}
