import { useState, useEffect } from "react";
import { LandingPage } from "./components/LandingPage";
import { AuthPage } from "./components/AuthPage";
import { Header } from "./components/Header";
import { TopNav } from "./components/TopNav";
import { Sidebar } from "./components/Sidebar";
import { WhatUseSection } from "./components/WhatUseSection";
import { BrandInfoSection } from "./components/BrandInfoSection";
import { ScoreSection } from "./components/ScoreSection";
import { motion, AnimatePresence } from "motion/react";

export type Section = "what-use" | "brand-info" | "score";

export default function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [showAuth, setShowAuth] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeSection, setActiveSection] = useState<Section>("what-use");

  // Check if user is already authenticated on mount
  useEffect(() => {
    const authStatus = localStorage.getItem("isAuthenticated");
    if (authStatus === "true") {
      setIsAuthenticated(true);
      // Don't automatically hide landing page - let user navigate
    }
  }, []);

  const handleNext = () => {
    const sections: Section[] = ["what-use", "brand-info", "score"];
    const currentIndex = sections.indexOf(activeSection);
    if (currentIndex < sections.length - 1) {
      setActiveSection(sections[currentIndex + 1]);
    }
  };

  const handleGetStarted = () => {
    // If already authenticated, go directly to form
    if (isAuthenticated) {
      setShowLanding(false);
      setShowAuth(false);
    } else {
      // Otherwise, show auth page
      setShowLanding(false);
      setShowAuth(true);
    }
  };

  const handleAuthSuccess = () => {
    setIsAuthenticated(true);
    setShowAuth(false);
    setShowLanding(false);
    setActiveSection("what-use");
  };

  const handleBackToLanding = () => {
    setShowLanding(true);
    setShowAuth(false);
    setIsAuthenticated(false);
    setActiveSection("what-use");
  };

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated");
    localStorage.removeItem("userEmail");
    localStorage.removeItem("userName");
    setIsAuthenticated(false);
    setShowLanding(true);
    setShowAuth(false);
    setActiveSection("what-use");
  };

  // Show landing page
  if (showLanding) {
    return <LandingPage onGetStarted={handleGetStarted} />;
  }

  // Show auth page
  if (showAuth && !isAuthenticated) {
    return (
      <AuthPage
        onAuthSuccess={handleAuthSuccess}
        onBackToLanding={handleBackToLanding}
      />
    );
  }

  // Show form sections (authenticated)
  if (!isAuthenticated) {
    return <LandingPage onGetStarted={handleGetStarted} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative">
      <div
        className="absolute inset-0 opacity-30 bg-cover bg-center"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1557682250-33bd709cbe85?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwdXJwbGUlMjBibHVlJTIwZ3JhZGllbnR8ZW58MXx8fHwxNzYzMDA0MzMwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
        }}
      />
      <div className="relative z-10">
        <Header onBackToLanding={handleBackToLanding} onLogout={handleLogout} />
        <TopNav activeSection={activeSection} />
        <div className="flex h-[calc(100vh-160px)]">
          <Sidebar
            activeSection={activeSection}
            onSectionChange={setActiveSection}
          />
          <main className="flex-1 p-12 overflow-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeSection}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {activeSection === "what-use" && (
                  <WhatUseSection onNext={handleNext} />
                )}
                {activeSection === "brand-info" && (
                  <BrandInfoSection onNext={handleNext} />
                )}
                {activeSection === "score" && <ScoreSection />}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </div>
  );
}
