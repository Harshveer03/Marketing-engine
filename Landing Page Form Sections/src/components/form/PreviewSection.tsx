import { useState } from "react";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Label } from "../ui/label";
import { Button } from "../ui/button";
import {
  Building2,
  Users,
  FileText,
  Palette,
  Edit2,
  Save,
  Eye,
  ArrowRight,
  TrendingUp,
  Globe,
} from "lucide-react";
import { motion } from "motion/react";

interface PreviewSectionProps {
  formData: {
    brandName: string;
    industry: string;
    description: string;
    targetAudience: string;
    targetIndustries?: string[];
    targetGeography?: string[];
  };
  onUpdateData: (data: {
    brandName: string;
    industry: string;
    description: string;
    targetAudience: string;
    targetIndustries?: string[];
    targetGeography?: string[];
  }) => void;
  onNext?: () => void;
}

export function PreviewSection({
  formData,
  onUpdateData,
  onNext,
}: PreviewSectionProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedData, setEditedData] = useState(formData);

  const handleChange = (field: string, value: string) => {
    setEditedData({
      ...editedData,
      [field]: value,
    });
  };

  const handleSave = () => {
    onUpdateData(editedData);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedData(formData);
    setIsEditing(false);
  };

  return (
    <div className="max-w-6xl mx-auto h-full flex flex-col">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center mb-8 py-5"
      >
        <motion.h2
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="text-5xl font-bold mb-4 text-gray-900"
        >
          Brand Information Preview
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-gray-700 text-xl font-semibold"
        >
          Review and edit your brand information
        </motion.p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white p-8 shadow-2xl hover:shadow-3xl transition-shadow duration-300 border-2 border-gray-200"
      >
        <div className="flex items-center justify-between mb-8 pb-6 border-b border-gray-200">
          <div className="flex items-center gap-4">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="w-14 h-14 !bg-black rounded-xl flex items-center justify-center shadow-xl"
            >
              <Eye className="w-6 h-6 text-black" />
            </motion.div>
            <h3 className="text-3xl text-gray-900 font-bold">
              Your Brand Details
            </h3>
          </div>

          {!isEditing ? (
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                onClick={() => setIsEditing(true)}
                className="!bg-black hover:!bg-gray-800 text-white border-0 shadow-xl hover:shadow-2xl rounded-xl px-6 py-3 font-semibold"
              >
                <Edit2 className="w-5 h-5 mr-2" />
                Edit Information
              </Button>
            </motion.div>
          ) : (
            <div className="flex gap-4">
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={handleCancel}
                  variant="outline"
                  className="border-2 border-gray-800 hover:border-black hover:bg-gray-50 text-gray-900 rounded-xl px-6 py-3 font-semibold"
                >
                  Cancel
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={handleSave}
                  className="!bg-black hover:!bg-gray-800 text-white border-0 shadow-xl hover:shadow-2xl rounded-xl px-6 py-3 font-semibold"
                >
                  <Save className="w-5 h-5 mr-2" />
                  Save Changes
                </Button>
              </motion.div>
            </div>
          )}
        </div>

        <div className="space-y-6">
          {/* Brand Name */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Label
              htmlFor="preview-brand-name"
              className="text-black font-semibold flex items-center gap-2 text-base mb-2"
            >
              <Building2 className="w-4 h-4 text-black" />
              Brand Name
            </Label>
            {isEditing ? (
              <Input
                id="preview-brand-name"
                value={editedData.brandName}
                onChange={(e) => handleChange("brandName", e.target.value)}
                className="border-2 border-gray-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all duration-200 h-11 text-base px-4"
                placeholder="Enter your brand name"
              />
            ) : (
              <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md">
                <p className="text-gray-800 text-base">
                  {formData.brandName || (
                    <span className="text-gray-400 italic">Not provided</span>
                  )}
                </p>
              </div>
            )}
          </motion.div>

          {/* Industry */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <Label
              htmlFor="preview-industry"
              className="text-black font-semibold flex items-center gap-2 text-base mb-2"
            >
              <Palette className="w-4 h-4 text-black" />
              Industry
            </Label>
            {isEditing ? (
              <Input
                id="preview-industry"
                value={editedData.industry}
                onChange={(e) => handleChange("industry", e.target.value)}
                className="border-2 border-gray-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all duration-200 h-11 text-base px-4"
                placeholder="e.g., Technology, Fashion, Food"
              />
            ) : (
              <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md">
                <p className="text-gray-800 text-base">
                  {formData.industry || (
                    <span className="text-gray-400 italic">Not provided</span>
                  )}
                </p>
              </div>
            )}
          </motion.div>

          {/* Brand Description */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.5 }}
          >
            <Label
              htmlFor="preview-description"
              className="text-black font-semibold flex items-center gap-2 text-base mb-2"
            >
              <FileText className="w-4 h-4 text-black" />
              Brand Description
            </Label>
            {isEditing ? (
              <Textarea
                id="preview-description"
                value={editedData.description}
                onChange={(e) => handleChange("description", e.target.value)}
                className="border-2 border-gray-200 min-h-[120px] focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all duration-200 resize-none text-base px-4 py-3"
                placeholder="Tell us about your brand, its mission, and values..."
              />
            ) : (
              <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md min-h-[120px]">
                <p className="text-gray-800 text-base whitespace-pre-wrap">
                  {formData.description || (
                    <span className="text-gray-400 italic">Not provided</span>
                  )}
                </p>
              </div>
            )}
          </motion.div>

          {/* Target Audience */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.6 }}
          >
            <Label
              htmlFor="preview-target-audience"
              className="text-black font-semibold flex items-center gap-2 text-base mb-2"
            >
              <Users className="w-4 h-4 text-black" />
              Target Audience
            </Label>
            {isEditing ? (
              <Input
                id="preview-target-audience"
                value={editedData.targetAudience}
                onChange={(e) => handleChange("targetAudience", e.target.value)}
                className="border-2 border-gray-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all duration-200 h-11 text-base px-4"
                placeholder="Who is your target audience?"
              />
            ) : (
              <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md">
                <p className="text-gray-800 text-base">
                  {formData.targetAudience || (
                    <span className="text-gray-400 italic">Not provided</span>
                  )}
                </p>
              </div>
            )}
          </motion.div>

          {/* Target Industries */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.7 }}
          >
            <Label className="text-black font-semibold flex items-center gap-2 text-base mb-2">
              <TrendingUp className="w-4 h-4 text-black" />
              Relevant Target Industries
            </Label>
            <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md min-h-[60px]">
              {formData.targetIndustries && formData.targetIndustries.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {formData.targetIndustries.map((industry, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 bg-gray-900 text-white px-3 py-1.5 rounded-full text-xs font-medium"
                    >
                      <span>{industry}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 italic text-base">Not provided</p>
              )}
            </div>
          </motion.div>

          {/* Target Geography */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.8 }}
          >
            <Label className="text-black font-semibold flex items-center gap-2 text-base mb-2">
              <Globe className="w-4 h-4 text-black" />
              Relevant Geography
            </Label>
            <div className="p-4 bg-gray-50 border-2 border-gray-200 rounded-md min-h-[60px]">
              {formData.targetGeography && formData.targetGeography.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {formData.targetGeography.map((geography, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 bg-gray-900 text-white px-3 py-1.5 rounded-full text-xs font-medium"
                    >
                      <span>{geography}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 italic text-base">Not provided</p>
              )}
            </div>
          </motion.div>
        </div>

        {/* Info Box */}
        {!isEditing && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="mt-8 p-6 bg-gray-50 border-2 border-gray-200 rounded-2xl"
          >
            <p className="text-gray-700 text-sm flex items-start gap-3">
              <span className="text-2xl">💡</span>
              <span>
                <strong className="font-bold">Pro Tip:</strong> Click the "Edit
                Information" button above to make changes to your brand
                information. Your changes will be saved automatically and synced
                across all sections.
              </span>
            </p>
          </motion.div>
        )}
      </motion.div>

      {/* Next Button */}
      {onNext && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="flex justify-end mt-8"
        >
          <motion.div
            whileHover={{ scale: 1.05, x: 5 }}
            whileTap={{ scale: 0.95 }}
          >
            <Button
              onClick={onNext}
              className="!bg-black hover:!bg-gray-800 text-white shadow-xl hover:shadow-2xl border-0 px-10 py-6 rounded-xl text-lg font-semibold"
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
      )}
    </div>
  );
}
