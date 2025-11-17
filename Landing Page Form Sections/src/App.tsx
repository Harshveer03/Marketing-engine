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

// Background images for each section
const sectionBackgrounds: Record<Section, string> = {
  "brand-info":
    "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1920&h=1080&fit=crop", // Office workspace
  preview:
    "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=1920&h=1080&fit=crop", // Business planning/documents
  score:
    "https://images.unsplash.com/photo-1543286386-713bdd548da4?w=1920&h=1080&fit=crop", // Success/achievement
  "what-use":
    "https://images.unsplash.com/photo-1553877522-43269d4ea984?w=1920&h=1080&fit=crop", // Creative/design workspace
};

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
    const sections: Section[] = ["brand-info", "preview", "score", "what-use"];
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
    <div className="h-screen bg-white relative flex flex-col overflow-hidden">
      {/* Background Image with transition */}
      <motion.div
        key={activeSection}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage: `url('${sectionBackgrounds[activeSection]}')`,
        }}
      />

      {/* White overlay for readability */}
      <div className="absolute inset-0 bg-white/90 z-0" />

      <div className="relative z-20 flex flex-col h-full min-h-0">
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
          <main className="flex-1 overflow-auto">
            {/* Progress Bar */}
            <div className="bg-white/95 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-40 px-12 py-4">
              <div className="max-w-6xl mx-auto relative">
                <div className="flex items-center gap-4 mb-3">
                  <span className="text-sm font-semibold text-gray-900 whitespace-nowrap">
                    Step{" "}
                    {["brand-info", "preview", "score", "what-use"].indexOf(
                      activeSection
                    ) + 1}{" "}
                    of 4
                  </span>

                  <div className="flex-1 h-2 bg-gray-300 rounded-full">
                    <div
                      style={{
                        width: `${
                          ([
                            "brand-info",
                            "preview",
                            "score",
                            "what-use",
                          ].indexOf(activeSection) +
                            1) *
                          25
                        }%`,
                        backgroundColor: "#000000",
                        height: "100%",
                        borderRadius: "9999px",
                        transition: "width 0.6s ease-in-out",
                      }}
                    />
                  </div>

                  <span className="text-sm font-semibold text-gray-900 whitespace-nowrap">
                    {(["brand-info", "preview", "score", "what-use"].indexOf(
                      activeSection
                    ) +
                      1) *
                      25}
                    % Complete
                  </span>
                </div>

                <div className="flex justify-between mt-3">
                  {[
                    { id: "brand-info", label: "Brand Info" },
                    { id: "preview", label: "Preview" },
                    { id: "score", label: "Score" },
                    { id: "what-use", label: "Usage" },
                  ].map((step, index) => {
                    const currentIndex = [
                      "brand-info",
                      "preview",
                      "score",
                      "what-use",
                    ].indexOf(activeSection);
                    const isCompleted = index < currentIndex;
                    const isCurrent = index === currentIndex;

                    return (
                      <div key={step.id} className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                            isCompleted || isCurrent
                              ? "bg-black"
                              : "bg-gray-300"
                          }`}
                        />
                        <span
                          className={`text-xs transition-colors duration-300 ${
                            isCurrent
                              ? "text-gray-900 font-semibold"
                              : "text-gray-500"
                          }`}
                        >
                          {step.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="p-12">
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
                      onNext={handleNext}
                    />
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </main>
        </div>
        <Footer />
      </div>
    </div>
  );
}
