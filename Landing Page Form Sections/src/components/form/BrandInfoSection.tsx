import { useState } from "react";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Label } from "../ui/label";
import { Button } from "../ui/button";
import {
  ArrowRight,
  Upload,
  FileText,
  Link as LinkIcon,
  Image,
  Mic,
  X,
  Globe,
  Users2,
  TrendingUp,
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
  // State for links
  const [websiteLinks, setWebsiteLinks] = useState<string[]>([]);
  const [websiteInput, setWebsiteInput] = useState("");

  // Social links - unified structure
  const [socialLinks, setSocialLinks] = useState<
    { platform: string; url: string }[]
  >([]);
  const [showSocialDropdown, setShowSocialDropdown] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = useState<string | null>(null);
  const [socialInput, setSocialInput] = useState("");

  const [competitorLinks, setCompetitorLinks] = useState<string[]>([]);
  const [competitorInput, setCompetitorInput] = useState("");
  const [showCompetitorInput, setShowCompetitorInput] = useState(false);

  const socialPlatforms = [
    { name: "Instagram", value: "instagram" },
    { name: "LinkedIn", value: "linkedin" },
    { name: "X (Twitter)", value: "twitter" },
    { name: "YouTube", value: "youtube" },
    { name: "Facebook", value: "facebook" },
  ];

  const handleChange = (field: string, value: string) => {
    onUpdateData({
      ...formData,
      [field]: value,
    });
  };

  const addLink = (
    input: string,
    setInput: (val: string) => void,
    links: string[],
    setLinks: (val: string[]) => void
  ) => {
    if (input.trim()) {
      setLinks([...links, input.trim()]);
      setInput("");
    }
  };

  const addSocialLink = () => {
    if (selectedPlatform && socialInput.trim()) {
      setSocialLinks([
        ...socialLinks,
        { platform: selectedPlatform, url: socialInput.trim() },
      ]);
      setSocialInput("");
      setSelectedPlatform(null);
      setShowSocialDropdown(false);
    }
  };

  const removeSocialLink = (index: number) => {
    setSocialLinks(socialLinks.filter((_, i) => i !== index));
  };

  const selectPlatform = (platform: string) => {
    setSelectedPlatform(platform);
    setShowSocialDropdown(false);
  };

  const removeLink = (
    index: number,
    links: string[],
    setLinks: (val: string[]) => void
  ) => {
    setLinks(links.filter((_, i) => i !== index));
  };

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

      {/* Single Unified Card - All Sections Combined */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="bg-white shadow-xl border-2 border-gray-200 p-8 overflow-auto max-h-[700px]"
      >
        <div className="space-y-8">
          {/* 1. Links & References Section */}
          <div>
            <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <LinkIcon className="w-5 h-5" />
              Links & References
            </h3>

            {/* Website Section - Full Width */}
            <div className="space-y-3 mb-6">
              <Label className="text-gray-900 font-semibold flex items-center gap-2 text-base">
                <Globe className="w-4 h-4" />
                Website
              </Label>
              <div className="flex gap-2">
                <Input
                  type="url"
                  value={websiteInput}
                  onChange={(e) => setWebsiteInput(e.target.value)}
                  placeholder="https://example.com"
                  className="flex-1 border-2 border-gray-300 h-10 focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 text-sm px-3 rounded-lg"
                />
                <Button
                  onClick={() =>
                    addLink(
                      websiteInput,
                      setWebsiteInput,
                      websiteLinks,
                      setWebsiteLinks
                    )
                  }
                  className="!bg-black hover:bg-gray-800 text-white px-4 border-0 text-xs h-10 font-semibold rounded-lg"
                >
                  Add
                </Button>
              </div>
              {websiteLinks.length > 0 && (
                <div className="space-y-2 mt-3">
                  {websiteLinks.map((link, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 bg-gray-50 p-2 rounded-lg"
                    >
                      <span className="text-xs text-gray-700 flex-1 truncate">
                        {link}
                      </span>
                      <button
                        onClick={() =>
                          removeLink(index, websiteLinks, setWebsiteLinks)
                        }
                        className="text-gray-500 hover:text-red-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Socials & Competitors - Buttons Row */}
            <div className="flex gap-4 pt-6">
              {/* Add Social Button with Dropdown */}
              <div className="flex-1 relative">
                <Button
                  onClick={() => setShowSocialDropdown(!showSocialDropdown)}
                  className="w-full !bg-black hover:bg-gray-800 text-white px-8 border-0 text-sm h-10 font-semibold rounded-lg flex items-center justify-center gap-2"
                >
                  + Add Social
                </Button>

                {/* Dropdown Menu */}
                {showSocialDropdown && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-white border-2 border-gray-300 rounded-lg shadow-lg z-10">
                    {socialPlatforms.map((platform) => (
                      <button
                        key={platform.value}
                        onClick={() => selectPlatform(platform.name)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 first:rounded-t-lg last:rounded-b-lg transition-colors"
                      >
                        {platform.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Add Competitor Button */}
              <div className="flex-1">
                <Button
                  onClick={() => setShowCompetitorInput(!showCompetitorInput)}
                  className="w-full !bg-black hover:bg-gray-800 text-white px-8 border-0 text-sm h-10 font-semibold rounded-lg flex items-center justify-center gap-2"
                >
                  + Add Competitor
                </Button>
              </div>
            </div>

            {/* Content Sections Below Buttons */}
            <div className="grid grid-cols-2 gap-6 mt-4">
              {/* Socials Content */}
              <div className="space-y-3">
                {/* Input field when platform is selected */}
                {selectedPlatform && (
                  <div className="space-y-2 p-3 bg-gray-50 rounded-lg">
                    <p className="text-xs font-medium text-gray-600">
                      {selectedPlatform}
                    </p>
                    <div className="flex gap-2">
                      <Input
                        type="url"
                        value={socialInput}
                        onChange={(e) => setSocialInput(e.target.value)}
                        placeholder={`${selectedPlatform} URL`}
                        className="flex-1 border-2 border-gray-300 h-9 focus:border-black text-xs px-3 rounded-lg"
                      />
                      <Button
                        onClick={addSocialLink}
                        className="!bg-black hover:bg-gray-800 text-white px-3 text-xs h-9 rounded-lg"
                      >
                        Add
                      </Button>
                      <Button
                        onClick={() => {
                          setSelectedPlatform(null);
                          setSocialInput("");
                        }}
                        className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-3 text-xs h-9 rounded-lg"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}

                {/* Display added social links */}
                {socialLinks.length > 0 && (
                  <div className="space-y-2">
                    {socialLinks.map((social, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-2 bg-gray-50 p-2 rounded-lg"
                      >
                        <div className="flex-1">
                          <p className="text-xs font-medium text-gray-600">
                            {social.platform}
                          </p>
                          <p className="text-xs text-gray-700 truncate">
                            {social.url}
                          </p>
                        </div>
                        <button
                          onClick={() => removeSocialLink(index)}
                          className="text-gray-500 hover:text-red-600"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Competitor Content */}
              <div className="space-y-3">
                {/* Input field when button is clicked */}
                {showCompetitorInput && (
                  <div className="space-y-2 p-3 bg-gray-50 rounded-lg">
                    <div className="flex gap-2">
                      <Input
                        type="url"
                        value={competitorInput}
                        onChange={(e) => setCompetitorInput(e.target.value)}
                        placeholder="Competitor URL"
                        className="flex-1 h-9 focus:border-black text-xs px-3 rounded-lg"
                      />
                      <Button
                        onClick={() => {
                          if (competitorInput.trim()) {
                            setCompetitorLinks([
                              ...competitorLinks,
                              competitorInput.trim(),
                            ]);
                            setCompetitorInput("");
                            setShowCompetitorInput(false);
                          }
                        }}
                        className="!bg-black hover:bg-gray-800 text-white px-3 text-xs h-9 rounded-lg"
                      >
                        Add
                      </Button>
                      <Button
                        onClick={() => {
                          setCompetitorInput("");
                          setShowCompetitorInput(false);
                        }}
                        className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-3 text-xs h-9 rounded-lg"
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}

                {/* Display added competitor links */}
                {competitorLinks.length > 0 && (
                  <div className="space-y-2 mt-3">
                    {competitorLinks.map((link, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-2 bg-gray-50 p-2 rounded-lg"
                      >
                        <span className="text-xs text-gray-700 flex-1 truncate">
                          {link}
                        </span>
                        <button
                          onClick={() =>
                            removeLink(
                              index,
                              competitorLinks,
                              setCompetitorLinks
                            )
                          }
                          className="text-gray-500 hover:text-red-600"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 2. Brand Assets Section */}
          <div className="pt-7">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Image className="w-5 h-5" />
              Brand Assets
            </h3>
            <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
              Logos, Images & Fonts
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
              <p className="text-gray-600 text-sm">PNG, JPG, SVG up to 10MB</p>
              <input
                type="file"
                className="absolute inset-0 opacity-0 cursor-pointer"
                multiple
                accept="image/*,.svg"
              />
            </motion.div>
          </div>

          {/* 3. Marketing Materials Section */}
          <div className="pt-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Upload className="w-5 h-6" />
              Marketing Materials
            </h3>
            <Label className="text-gray-900 font-semibold flex items-center gap-2 mb-3 text-base">
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
              <p className="text-gray-600 text-sm">PDF, DOC, PPT up to 20MB</p>
              <input
                type="file"
                className="absolute inset-0 opacity-0 cursor-pointer"
                multiple
                accept=".pdf,.doc,.docx,.ppt,.pptx"
              />
            </motion.div>
          </div>

          {/* 4. Brand Description Section */}
          <div className="pt-8">
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <FileText className="w-5 h-6" />
              Brand Description
            </h3>
            <div>
              <Label
                htmlFor="description"
                className="text-gray-900 font-semibold flex items-center gap-2 text-base mb-2"
              >
                <FileText className="w-4 h-4 text-gray-700" />
                Tell us about your brand
              </Label>
              <div className="flex gap-2">
                <Textarea
                  id="description"
                  value={formData.description}
                  onChange={(e) => handleChange("description", e.target.value)}
                  className="flex-1 border-2 border-gray-300 min-h-[120px] focus:border-black focus:ring-2 focus:ring-gray-200 transition-all duration-200 resize-none text-base px-4 py-3 rounded-lg"
                  placeholder="Tell us about your brand, its mission, and values..."
                />
                <Button
                  type="button"
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-2 border-gray-300 px-4 h-11 rounded-lg self-start"
                >
                  <Mic className="w-5 h-5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex justify-end mt-12 h-20 items-center shrink-0"
      >
        <motion.div
          whileHover={{ scale: 1.05, x: 5 }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            onClick={onNext}
            className="!bg-black hover:bg-gray-800 text-white shadow-xl hover:shadow-2xl border-0 px-10 py-6 rounded-xl text-lg font-semibold"
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
