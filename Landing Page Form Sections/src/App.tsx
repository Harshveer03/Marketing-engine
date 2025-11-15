import { useState, useEffect } from "react";
import { LandingPage } from "./components/LandingPage";
import { AuthPage } from "./components/AuthPage";
import { Header } from "./components/Header";
import { Footer } from "./components/Footer";
import { Sidebar } from "./components/Sidebar";
import { WhatUseSection } from "./components/WhatUseSection";
import { BrandInfoSection } from "./components/BrandInfoSection";
import { ScoreSection } from "./components/ScoreSection";
import { PreviewSection } from "./components/PreviewSection";
import { SplashScreen } from "./components/SplashScreen";
import { motion, AnimatePresence } from "motion/react";

export type Section = "brand-info" | "score" | "what-use" | "preview";

// Form data types
interface FormData {
  whatUse: string | null;
  brandInfo: {
    brandName: string;
    industry: string;
    description: string;
    targetAudience: string;
  };
}

export default function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [showLanding, setShowLanding] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [activeSection, setActiveSection] = useState<Section>("brand-info");

  // Form data state
  const [formData, setFormData] = useState<FormData>({
    whatUse: null,
    brandInfo: {
      brandName: "",
      industry: "",
      description: "",
      targetAudience: "",
    },
  });

  // Check if user is already authenticated on mount and restore state
  useEffect(() => {
    const authStatus = localStorage.getItem("isAuthenticated");
    if (authStatus === "true") {
      setIsAuthenticated(true);

      // Restore active section
      const savedSection = localStorage.getItem("activeSection") as Section;
      if (savedSection) {
        setActiveSection(savedSection);
        setShowLanding(false);
      }

      // Restore form data
      const savedFormData = localStorage.getItem("formData");
      if (savedFormData) {
        try {
          setFormData(JSON.parse(savedFormData));
        } catch (e) {
          console.error("Failed to parse saved form data", e);
        }
      }
    }
  }, []);

  // Save active section to localStorage whenever it changes
  useEffect(() => {
    if (isAuthenticated && !showLanding) {
      localStorage.setItem("activeSection", activeSection);
    }
  }, [activeSection, isAuthenticated, showLanding]);

  // Save form data to localStorage whenever it changes
  useEffect(() => {
    if (isAuthenticated) {
      localStorage.setItem("formData", JSON.stringify(formData));
    }
  }, [formData, isAuthenticated]);

  const handleNext = () => {
    const sections: Section[] = ["brand-info", "score", "what-use", "preview"];
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
    setActiveSection("brand-info");
  };

  const handleBackToLanding = () => {
    setShowLanding(true);
    setShowAuth(false);
    setIsAuthenticated(false);
    setActiveSection("brand-info");
  };

  const handleLogout = () => {
    localStorage.removeItem("isAuthenticated");
    localStorage.removeItem("userEmail");
    localStorage.removeItem("userName");
    // Don't clear form data on logout - only on successful submission
    setIsAuthenticated(false);
    setShowLanding(true);
    setShowAuth(false);
    setActiveSection("brand-info");
  };

  const handleClearForm = () => {
    const emptyFormData: FormData = {
      whatUse: null,
      brandInfo: {
        brandName: "",
        industry: "",
        description: "",
        targetAudience: "",
      },
    };
    setFormData(emptyFormData);
    localStorage.setItem("formData", JSON.stringify(emptyFormData));
    setActiveSection("brand-info");
  };

  const updateFormData = (section: keyof FormData, data: any) => {
    setFormData((prev) => ({
      ...prev,
      [section]: data,
    }));
  };

  // Show landing page
  if (showLanding) {
    return <LandingPage onGetStarted={handleGetStarted} />;
  }

  // Show splash screen first
  if (showSplash) {
    return (
      <SplashScreen
        onComplete={() => {
          setShowSplash(false);
          setShowLanding(true);
        }}
      />
    );
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
    <div className="h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative flex flex-col overflow-hidden">
      <div
        className="absolute inset-0 opacity-30 bg-cover bg-center"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1557682250-33bd709cbe85?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwdXJwbGUlMjBibHVlJTIwZ3JhZGllbnR8ZW58MXx8fHwxNzYzMDA0MzMwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
        }}
      />
      <div className="relative z-10 flex flex-col h-full min-h-0">
        <Header 
          onBackToLanding={handleBackToLanding} 
          onLogout={handleLogout}
          onClearForm={handleClearForm}
        />
        <div className="flex flex-1 overflow-hidden">
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
                className="h-full"
              >
                {activeSection === "brand-info" && (
                  <BrandInfoSection
                    onNext={handleNext}
                    formData={formData.brandInfo}
                    onUpdateData={(data) => updateFormData("brandInfo", data)}
                  />
                )}
                {activeSection === "score" && <ScoreSection />}
                {activeSection === "what-use" && (
                  <WhatUseSection
                    onNext={handleNext}
                    selectedOption={formData.whatUse}
                    onSelectOption={(option) =>
                      updateFormData("whatUse", option)
                    }
                  />
                )}
                {activeSection === "preview" && (
                  <PreviewSection
                    formData={formData.brandInfo}
                    onUpdateData={(data) => updateFormData("brandInfo", data)}
                  />
                )}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
        <Footer />
      </div>
    </div>
  );
}
