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
} from "lucide-react";
import { motion } from "motion/react";

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
  const handleChange = (field: string, value: string) => {
    onUpdateData({
      ...formData,
      [field]: value,
    });
  };
  return (
    <div className="max-w-6xl mx-auto px-4 h-full flex flex-col">
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

      <div className="grid grid-cols-2 gap-3">
        {/* Left Column - Brand Info */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <div className="bg-white p-8 rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 border-2 border-gray-200 h-full">
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-200">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-12 h-12 bg-black rounded-xl flex items-center justify-center shadow-lg"
              >
                <Building2 className="w-6 h-6 text-white" />
              </motion.div>
              <h3 className="text-2xl font-bold text-gray-900">
                Basic Information
              </h3>
            </div>

            <div className="space-y-4 flex flex-col h-[calc(100%-3.5rem)]">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.3 }}
              >
                <Label
                  htmlFor="brand-name"
                  className="text-gray-900 font-semibold flex items-center gap-2 text-base"
                >
                  <Building2 className="w-4 h-4 text-gray-700" />
                  Brand Name
                </Label>
                <Input
                  id="brand-name"
                  value={formData.brandName}
                  onChange={(e) => handleChange("brandName", e.target.value)}
                  className="border-2 border-gray-300 mt-2 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-12 text-base px-4 rounded-xl"
                  placeholder="Enter your brand name"
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.4 }}
              >
                <Label
                  htmlFor="industry"
                  className="text-gray-900 font-semibold flex items-center gap-2 text-base"
                >
                  <Palette className="w-4 h-4 text-gray-700" />
                  Industry
                </Label>
                <Input
                  id="industry"
                  value={formData.industry}
                  onChange={(e) => handleChange("industry", e.target.value)}
                  className="border-2 border-gray-300 mt-2 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-12 text-base px-4 rounded-xl"
                  placeholder="e.g., Technology, Fashion, Food"
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.5 }}
                className="flex-1"
              >
                <Label
                  htmlFor="description"
                  className="text-gray-900 font-semibold flex items-center gap-2 text-base"
                >
                  <FileText className="w-4 h-4 text-gray-700" />
                  Brand Description
                </Label>
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => handleChange("description", e.target.value)}
                  className="border-2 border-gray-300 mt-2 h-full min-h-[100px] focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 resize-none text-base px-4 py-3 rounded-xl"
                  placeholder="Tell us about your brand, its mission, and values..."
                />
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.6 }}
              >
                <Label
                  htmlFor="target-audience"
                  className="text-gray-900 font-semibold flex items-center gap-2 text-base"
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
                  className="border-2 border-gray-300 mt-2 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 h-12 text-base px-4 rounded-xl"
                  placeholder="Who is your target audience?"
                />
              </motion.div>
            </div>
          </div>
        </motion.div>

        {/* Right Column - Resources */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <div className="bg-white p-8 rounded-2xl shadow-xl hover:shadow-2xl transition-shadow duration-300 border-2 border-gray-200 h-full">
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-200">
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
                className="w-12 h-12 bg-black rounded-xl flex items-center justify-center shadow-lg"
              >
                <FileText className="w-6 h-6 text-white" />
              </motion.div>
              <h3 className="text-2xl font-bold text-gray-900">Resources</h3>
            </div>

            <div className="space-y-4">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
              >
                <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-base">
                  <Image className="w-4 h-4 text-gray-700" />
                  Brand Assets
                </Label>
                <motion.div
                  whileHover={{ scale: 1.01 }}
                  className="group relative bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-5 text-center hover:border-black hover:bg-gray-100 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center"
                >
                  <div className="w-12 h-12 mx-auto mb-3 bg-white rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow border border-gray-200">
                    <Image className="w-6 h-6 text-gray-700" />
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
                  />
                </motion.div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.4 }}
              >
                <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-base">
                  <FileText className="w-4 h-4 text-gray-700" />
                  Existing Materials
                </Label>
                <motion.div
                  whileHover={{ scale: 1.01 }}
                  className="group relative bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-5 text-center hover:border-black hover:bg-gray-100 transition-all duration-300 cursor-pointer flex flex-col items-center justify-center"
                >
                  <div className="w-12 h-12 mx-auto mb-3 bg-white rounded-lg flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow border border-gray-200">
                    <Upload className="w-6 h-6 text-gray-700" />
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
                  />
                </motion.div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
              >
                <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-base">
                  <LinkIcon className="w-4 h-4 text-gray-700" />
                  Reference Links
                </Label>
                <div className="p-4">
                  <div className="flex gap-3">
                    <Input
                      type="url"
                      placeholder="https://example.com"
                      className="flex-1 border-2 border-gray-300 h-12 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 text-base px-4 rounded-xl"
                    />
                    <Button className="bg-black hover:bg-gray-800 text-white px-8 border-0 shadow-md text-base h-12 font-semibold rounded-xl">
                      Add
                    </Button>
                  </div>
                  <p className="text-gray-600 text-sm mt-2">
                    Add your website link
                  </p>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex justify-end mt-4"
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
