import { useState } from "react";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import {
  ArrowRight,
  Upload,
  Building2,
  Users,
  FileText,
  Link as LinkIcon,
  Image,
  Palette,
  Mic,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

type TabType = "text" | "documents" | "links" | "audio";

interface BrandInfoSectionProps {
  onNext: () => void;
  formData: {
    brandName: string;
    industry: string;
    description: string;
    targetAudience: string;
  };
  onUpdateData: (data: {
    brandName: string;
    industry: string;
    description: string;
    targetAudience: string;
  }) => void;
}

export function BrandInfoSection({
  onNext,
  formData,
  onUpdateData,
}: BrandInfoSectionProps) {
  const [activeTab, setActiveTab] = useState<TabType>("text");

  const handleChange = (field: string, value: string) => {
    onUpdateData({
      ...formData,
      [field]: value,
    });
  };

  const tabs = [
    { id: "text" as TabType, label: "Text", icon: FileText },
    { id: "documents" as TabType, label: "Documents", icon: Upload },
    { id: "links" as TabType, label: "Reference Links", icon: LinkIcon },
    { id: "audio" as TabType, label: "Audio", icon: Mic },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 flex flex-col">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-10 py-4"
      >
        <motion.h2
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-bold mb-4 text-white drop-shadow-[0_2px_8px_rgba(0,0,0,0.8)]"
          style={{ textShadow: "2px 2px 4px rgba(0,0,0,0.9)" }}
        >
          Brand Information & Resources
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-white text-xl font-semibold drop-shadow-[0_2px_6px_rgba(0,0,0,0.8)]"
          style={{ textShadow: "1px 1px 3px rgba(0,0,0,0.9)" }}
        >
          Tell us about your brand and upload your assets
        </motion.p>
      </motion.div>

      {/* Single Unified Card with Tabs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white shadow-xl border-2 border-gray-200 h-[600px] flex flex-col"
      >
        {/* Tab Navigation */}
        <div className="border-b border-gray-200 shrink-0">
          <div className="flex">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 px-6 py-4 flex items-center justify-center gap-2 font-semibold transition-all duration-200 relative ${
                    isActive
                      ? "text-black bg-gray-50"
                      : "text-gray-600 hover:text-black hover:bg-gray-50"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-black"
                      transition={{
                        type: "spring",
                        stiffness: 300,
                        damping: 30,
                      }}
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Tab Content */}
        <div className="p-8 flex-1 overflow-auto">
          <AnimatePresence mode="wait">
            {activeTab === "text" && (
              <motion.div
                key="text"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                <div>
                  <Label
                    htmlFor="brand-name"
                    className="text-gray-900 font-semibold flex items-center gap-2 text-base mb-2"
                  >
                    <Building2 className="w-4 h-4 text-gray-700" />
                    Brand Name
                  </Label>
                  <Input
                    id="brand-name"
                    value={formData.brandName}
                    onChange={(e) => handleChange("brandName", e.target.value)}
                    className="border-2 border-gray-300 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-11 text-base px-4 rounded-lg"
                    placeholder="Enter your brand name"
                  />
                </div>

                <div>
                  <Label
                    htmlFor="industry"
                    className="text-gray-900 font-semibold flex items-center gap-2 text-base mb-2"
                  >
                    <Palette className="w-4 h-4 text-gray-700" />
                    Industry
                  </Label>
                  <Input
                    id="industry"
                    value={formData.industry}
                    onChange={(e) => handleChange("industry", e.target.value)}
                    className="border-2 border-gray-300 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-11 text-base px-4 rounded-lg"
                    placeholder="e.g., Technology, Fashion, Food"
                  />
                </div>

                <div>
                  <Label
                    htmlFor="description"
                    className="text-gray-900 font-semibold flex items-center gap-2 text-base mb-2"
                  >
                    <FileText className="w-4 h-4 text-gray-700" />
                    Brand Description
                  </Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) =>
                      handleChange("description", e.target.value)
                    }
                    className="border-2 border-gray-300 min-h-[120px] focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 resize-none text-base px-4 py-3 rounded-lg"
                    placeholder="Tell us about your brand, its mission, and values..."
                  />
                </div>

                <div>
                  <Label
                    htmlFor="target-audience"
                    className="text-gray-900 font-semibold flex items-center gap-2 text-base mb-2"
                  >
                    <Users className="w-4 h-4 text-gray-700" />
                    Target Audience
                  </Label>
                  <Input
                    id="target-audience"
                    value={formData.targetAudience}
                    onChange={(e) =>
                      handleChange("targetAudience", e.target.value)
                    }
                    className="border-2 border-gray-300 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-11 text-base px-4 rounded-lg"
                    placeholder="Who is your target audience?"
                  />
                </div>
              </motion.div>
            )}

            {activeTab === "documents" && (
              <motion.div
                key="documents"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                <div>
                  <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
                    <Image className="w-4 h-4 text-gray-700" />
                    Brand Assets
                  </Label>
                  <motion.div
                    whileHover={{ scale: 1.01 }}
                    className="group relative bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-black hover:bg-gray-100 transition-all duration-300 cursor-pointer"
                  >
                    <div className="w-14 h-14 mx-auto mb-3 bg-white rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow border border-gray-200">
                      <Image className="w-7 h-7 text-gray-700" />
                    </div>
                    <p className="text-gray-900 font-semibold text-sm mb-1">
                      Upload logos, images, fonts
                    </p>
                    <p className="text-gray-600 text-sm">
                      PNG, JPG, SVG up to 10MB
                    </p>
                    <input
                      type="file"
                      className="absolute inset-0 opacity-0 cursor-pointer"
                      multiple
                      accept="image/*,.svg"
                    />
                  </motion.div>
                </div>

                <div>
                  <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
                    <FileText className="w-4 h-4 text-gray-700" />
                    Existing Materials
                  </Label>
                  <motion.div
                    whileHover={{ scale: 1.01 }}
                    className="group relative bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-black hover:bg-gray-100 transition-all duration-300 cursor-pointer"
                  >
                    <div className="w-14 h-14 mx-auto mb-3 bg-white rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow border border-gray-200">
                      <Upload className="w-7 h-7 text-gray-700" />
                    </div>
                    <p className="text-gray-900 font-semibold text-sm mb-1">
                      Upload marketing materials
                    </p>
                    <p className="text-gray-600 text-sm">
                      PDF, DOC, PPT up to 20MB
                    </p>
                    <input
                      type="file"
                      className="absolute inset-0 opacity-0 cursor-pointer"
                      multiple
                      accept=".pdf,.doc,.docx,.ppt,.pptx"
                    />
                  </motion.div>
                </div>
              </motion.div>
            )}

            {activeTab === "links" && (
              <motion.div
                key="links"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
                  <LinkIcon className="w-4 h-4 text-gray-700" />
                  Reference Links
                </Label>
                <div className="space-y-4">
                  <div className="flex gap-3">
                    <Input
                      type="url"
                      placeholder="https://example.com"
                      className="flex-1 border-2 border-gray-300 h-11 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 text-base px-4 rounded-lg"
                    />
                    <Button className="bg-black hover:bg-gray-800 text-white px-6 border-0 shadow-md text-sm h-11 font-semibold rounded-lg">
                      Add
                    </Button>
                  </div>
                  <p className="text-gray-600 text-sm">
                    Add your website link or any reference URLs
                  </p>
                </div>
              </motion.div>
            )}

            {activeTab === "audio" && (
              <motion.div
                key="audio"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
                  <Mic className="w-4 h-4 text-gray-700" />
                  Audio Files
                </Label>
                <motion.div
                  whileHover={{ scale: 1.01 }}
                  className="group relative bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-black hover:bg-gray-100 transition-all duration-300 cursor-pointer"
                >
                  <div className="w-16 h-16 mx-auto mb-4 bg-white rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow border border-gray-200">
                    <Mic className="w-8 h-8 text-gray-700" />
                  </div>
                  <p className="text-gray-900 font-semibold text-base mb-2">
                    Upload audio files
                  </p>
                  <p className="text-gray-600 text-sm">
                    MP3, WAV, M4A up to 50MB
                  </p>
                  <input
                    type="file"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    multiple
                    accept="audio/*"
                  />
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex justify-end mt-6 h-20 items-center shrink-0"
      >
        <motion.div
          whileHover={{ scale: 1.05, x: 5 }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            onClick={onNext}
            className="bg-black hover:bg-gray-800 text-white shadow-xl hover:shadow-2xl border-0 px-10 py-6 rounded-xl text-lg font-semibold"
          >
            <span className="flex items-center gap-3">
              Continue to Next Step
              <motion.div
                animate={{ x: [0, 5, 0] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <ArrowRight className="w-5 h-5" />
              </motion.div>
            </span>
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
