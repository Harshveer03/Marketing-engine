import { useState } from "react";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import {
  Building2,
  Users,
  FileText,
  Palette,
  Edit2,
  Save,
  Eye,
} from "lucide-react";
import { motion } from "motion/react";

interface PreviewSectionProps {
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

export function PreviewSection({
  formData,
  onUpdateData,
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
        className="text-center mb-8 py-6"
      >
        <motion.h2 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="text-4xl font-bold mb-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent"
        >
          Brand Information Preview
        </motion.h2>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-gray-600 text-lg"
        >
          Review and edit your brand information
        </motion.p>
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="w-24 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 mx-auto mt-4 rounded-full"
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white p-10 shadow-2xl hover:shadow-3xl transition-shadow duration-300 border-2 border-indigo-100 rounded-3xl"
      >
        <div className="flex items-center justify-between mb-8 pb-6 border-b border-indigo-100">
          <div className="flex items-center gap-4">
            <motion.div 
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="w-14 h-14 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-xl"
            >
              <Eye className="w-7 h-7 text-white" />
            </motion.div>
            <h3 className="text-3xl text-gray-800 font-bold">
              Your Brand Details
            </h3>
          </div>

          {!isEditing ? (
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                onClick={() => setIsEditing(true)}
                className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 text-white border-0 shadow-xl hover:shadow-2xl rounded-xl px-6 py-3 font-semibold"
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
                  className="border-2 border-gray-300 hover:border-gray-400 rounded-xl px-6 py-3 font-semibold"
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
                  className="bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 hover:from-green-600 hover:via-emerald-600 hover:to-teal-600 text-white border-0 shadow-xl hover:shadow-2xl rounded-xl px-6 py-3 font-semibold"
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
              className="text-gray-700 font-medium flex items-center gap-2 text-base mb-2"
            >
              <Building2 className="w-4 h-4 text-indigo-500" />
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
              className="text-gray-700 font-medium flex items-center gap-2 text-base mb-2"
            >
              <Palette className="w-4 h-4 text-purple-500" />
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
              className="text-gray-700 font-medium flex items-center gap-2 text-base mb-2"
            >
              <FileText className="w-4 h-4 text-indigo-500" />
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
              className="text-gray-700 font-medium flex items-center gap-2 text-base mb-2"
            >
              <Users className="w-4 h-4 text-purple-500" />
              Target Audience
            </Label>
            {isEditing ? (
              <Input
                id="preview-target-audience"
                value={editedData.targetAudience}
                onChange={(e) =>
                  handleChange("targetAudience", e.target.value)
                }
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
        </div>

        {/* Info Box */}
        {!isEditing && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="mt-8 p-5 bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-2xl"
          >
            <p className="text-indigo-700 text-sm flex items-start gap-3">
              <span className="text-2xl">💡</span>
              <span>
                <strong className="font-bold">Pro Tip:</strong> Click the "Edit Information" button above to make
                changes to your brand information. Your changes will be saved
                automatically and synced across all sections.
              </span>
            </p>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
