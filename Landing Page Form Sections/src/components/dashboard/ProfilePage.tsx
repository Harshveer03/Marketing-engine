import { useState, useRef } from "react";
import { motion } from "motion/react";
import {
  User,
  Mail,
  Phone,
  MapPin,
  Building2,
  Globe,
  Camera,
  Edit2,
  Save,
  X,
  Shield,
  Bell,
  Palette,
  CreditCard,
  FileText,
  TrendingUp,
  Upload,
  Trash2,
  Download,
} from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";

export function ProfilePage() {
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    name: "John Doe",
    email: "john.doe@example.com",
    phone: "+1 (555) 123-4567",
    location: "San Francisco, CA",
    company: "Acme Inc.",
    website: "www.acme.com",
    bio: "Marketing professional with 10+ years of experience in digital strategy and brand development.",
  });

  const [editedData, setEditedData] = useState(profileData);

  const handleSave = () => {
    setProfileData(editedData);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedData(profileData);
    setIsEditing(false);
  };

  const handleChange = (field: string, value: string) => {
    setEditedData({
      ...editedData,
      [field]: value,
    });
  };

  // Brand Information State
  const [isEditingBrand, setIsEditingBrand] = useState(false);
  const [brandData, setBrandData] = useState({
    brandName: "Acme Inc.",
    industry: "Technology & Software",
    description: "We create innovative solutions for modern businesses.",
    targetAudience: "Small to medium-sized businesses",
    targetIndustries: ["Technology & Software", "E-commerce & Retail"],
    targetGeography: ["North America", "Europe"],
  });
  const [editedBrandData, setEditedBrandData] = useState(brandData);

  const handleBrandSave = () => {
    setBrandData(editedBrandData);
    setIsEditingBrand(false);
  };

  const handleBrandCancel = () => {
    setEditedBrandData(brandData);
    setIsEditingBrand(false);
  };

  const handleBrandChange = (field: string, value: string) => {
    setEditedBrandData({
      ...editedBrandData,
      [field]: value,
    });
  };

  // Helper functions for managing industries and geography
  const addIndustry = (industry: string) => {
    if (!editedBrandData.targetIndustries.includes(industry) && editedBrandData.targetIndustries.length < 5) {
      setEditedBrandData({
        ...editedBrandData,
        targetIndustries: [...editedBrandData.targetIndustries, industry],
      });
    }
  };

  const removeIndustry = (index: number) => {
    setEditedBrandData({
      ...editedBrandData,
      targetIndustries: editedBrandData.targetIndustries.filter((_, i) => i !== index),
    });
  };

  const addGeography = (geography: string) => {
    if (!editedBrandData.targetGeography.includes(geography) && editedBrandData.targetGeography.length < 5) {
      setEditedBrandData({
        ...editedBrandData,
        targetGeography: [...editedBrandData.targetGeography, geography],
      });
    }
  };

  const removeGeography = (index: number) => {
    setEditedBrandData({
      ...editedBrandData,
      targetGeography: editedBrandData.targetGeography.filter((_, i) => i !== index),
    });
  };

  // Predefined options
  const commonIndustries = [
    "Technology & Software",
    "Healthcare & Medical",
    "Finance & Banking",
    "E-commerce & Retail",
    "Education & Training",
    "Real Estate",
    "Food & Beverage",
    "Travel & Hospitality",
    "Manufacturing",
    "Professional Services",
    "Entertainment & Media",
    "Fashion & Apparel",
    "Automotive",
    "Energy & Utilities",
    "Non-Profit & NGO",
  ];

  const geographyOptions = [
    "Global",
    "North America",
    "United States",
    "Canada",
    "Mexico",
    "Europe",
    "United Kingdom",
    "Germany",
    "France",
    "Asia-Pacific",
    "China",
    "India",
    "Japan",
    "Australia",
    "Middle East",
    "Africa",
    "Latin America",
    "South America",
  ];

  // Uploaded Files State
  const [uploadedFiles, setUploadedFiles] = useState([
    { id: 1, name: "Brand_Logo.png", type: "Logo", size: "245 KB", date: "2024-11-20" },
    { id: 2, name: "Marketing_Deck.pdf", type: "Document", size: "1.2 MB", date: "2024-11-18" },
    { id: 3, name: "Product_Images.zip", type: "Images", size: "5.8 MB", date: "2024-11-15" },
  ]);

  const handleFileDelete = (id: number) => {
    setUploadedFiles(uploadedFiles.filter(file => file.id !== id));
  };

  // File upload functionality
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const newFiles = Array.from(files).map((file, index) => {
        const fileType = file.type.includes('image') ? 'Image' : 
                        file.type.includes('pdf') ? 'Document' : 
                        file.type.includes('zip') ? 'Archive' : 'File';
        const fileSize = file.size < 1024 * 1024 
          ? `${Math.round(file.size / 1024)} KB` 
          : `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
        
        return {
          id: uploadedFiles.length + index + 1,
          name: file.name,
          type: fileType,
          size: fileSize,
          date: new Date().toISOString().split('T')[0],
        };
      });
      
      setUploadedFiles([...uploadedFiles, ...newFiles]);
      // Reset the input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="p-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 max-w-7xl mx-auto"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Profile</h1>
        <p className="text-gray-600">Manage your account settings and preferences</p>
      </motion.div>

      {/* Profile Summary Card - Full Width */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="max-w-7xl mx-auto mb-8"
      >
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
          <div className="flex items-center gap-8">
            {/* Profile Picture */}
            <div className="relative flex-shrink-0">
              <div className="w-32 h-32 bg-gradient-to-br from-gray-900 to-gray-700 rounded-full flex items-center justify-center text-black text-4xl font-bold">
                <User className="w-6 h-6" />              
                </div>
            </div>

            {/* Name and Basic Info */}
            <div className="flex-1">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                {profileData.name}
              </h2>
              <p className="text-gray-600 mb-4">{profileData.email}</p>
              <div className="flex gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <FileText className="w-5 h-5 text-gray-700" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">127</p>
                    <p className="text-xs text-gray-600">Posts</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-gray-700" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">23</p>
                    <p className="text-xs text-gray-600">Campaigns</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                    <CreditCard className="w-5 h-5 text-gray-700" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-gray-900">25</p>
                    <p className="text-xs text-gray-600">Credits</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 flex-shrink-0">
              <Button className="!bg-black hover:bg-gray-800 text-white">
                <CreditCard className="w-4 h-4 mr-2" />
                Manage Subscription
              </Button>
              <Button
                variant="outline"
                className="border-2 border-gray-300 hover:bg-gray-50"
              >
                <Shield className="w-4 h-4 mr-2" />
                Privacy
              </Button>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column - Personal & Brand Information */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-8"
        >
          {/* Personal Information Card */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-8">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  Personal Information
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  Update your personal details and contact information
                </p>
              </div>
              {!isEditing ? (
                <Button
                  onClick={() => setIsEditing(true)}
                  className="!bg-black hover:bg-gray-800 text-white"
                >
                  <Edit2 className="w-4 h-4 mr-2" />
                  Edit
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    className="border-2 border-gray-300"
                  >
                    <X className="w-4 h-4 mr-2" />
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSave}
                    className="!bg-black hover:bg-gray-800 text-white"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                </div>
              )}
            </div>

            {/* Contact Information Section */}
            <div className="mb-8">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
                Contact Information
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Full Name */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <User className="w-4 h-4 text-gray-500" />
                    Full Name
                  </Label>
                  {isEditing ? (
                    <Input
                      value={editedData.name}
                      onChange={(e) => handleChange("name", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="Enter your full name"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.name}</p>
                    </div>
                  )}
                </div>

                {/* Email */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Mail className="w-4 h-4 text-gray-500" />
                    Email Address
                  </Label>
                  {isEditing ? (
                    <Input
                      type="email"
                      value={editedData.email}
                      onChange={(e) => handleChange("email", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="your.email@example.com"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.email}</p>
                    </div>
                  )}
                </div>

                {/* Phone */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Phone className="w-4 h-4 text-gray-500" />
                    Phone Number
                  </Label>
                  {isEditing ? (
                    <Input
                      type="tel"
                      value={editedData.phone}
                      onChange={(e) => handleChange("phone", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="+1 (555) 000-0000"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.phone}</p>
                    </div>
                  )}
                </div>

                {/* Location */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <MapPin className="w-4 h-4 text-gray-500" />
                    Location
                  </Label>
                  {isEditing ? (
                    <Input
                      value={editedData.location}
                      onChange={(e) => handleChange("location", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="City, State/Country"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.location}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Professional Information Section */}
            <div className="mb-8">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
                Professional Information
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Company */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Building2 className="w-4 h-4 text-gray-500" />
                    Company
                  </Label>
                  {isEditing ? (
                    <Input
                      value={editedData.company}
                      onChange={(e) => handleChange("company", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="Your company name"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.company}</p>
                    </div>
                  )}
                </div>

                {/* Website */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Globe className="w-4 h-4 text-gray-500" />
                    Website
                  </Label>
                  {isEditing ? (
                    <Input
                      value={editedData.website}
                      onChange={(e) => handleChange("website", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="www.yourwebsite.com"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{profileData.website}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* About Section */}
            <div>
              <h4 className="text-lg font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
                About
              </h4>
              <div>
                <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                  <FileText className="w-4 h-4 text-gray-500" />
                  Bio
                </Label>
                {isEditing ? (
                  <textarea
                    value={editedData.bio}
                    onChange={(e) => handleChange("bio", e.target.value)}
                    className="w-full p-4 border-2 border-gray-300 focus:border-black rounded-lg min-h-[120px] resize-none text-gray-900"
                    placeholder="Tell us about yourself, your role, and your expertise..."
                  />
                ) : (
                  <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <p className="text-gray-900 leading-relaxed">{profileData.bio}</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Brand Information Card */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-8">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  Brand Information
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  Manage your brand details and target market
                </p>
              </div>
              {!isEditingBrand ? (
                <Button
                  onClick={() => setIsEditingBrand(true)}
                  className="!bg-black hover:bg-gray-800 text-white"
                >
                  <Edit2 className="w-4 h-4 mr-2" />
                  Edit
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    onClick={handleBrandCancel}
                    variant="outline"
                    className="border-2 border-gray-300"
                  >
                    <X className="w-4 h-4 mr-2" />
                    Cancel
                  </Button>
                  <Button
                    onClick={handleBrandSave}
                    className="!bg-black hover:bg-gray-800 text-white"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                </div>
              )}
            </div>

            {/* Basic Brand Information Section */}
            <div className="mb-8">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
                Basic Information
              </h4>
              <div className="space-y-6">
                {/* Brand Name */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Building2 className="w-4 h-4 text-gray-500" />
                    Brand Name
                  </Label>
                  {isEditingBrand ? (
                    <Input
                      value={editedBrandData.brandName}
                      onChange={(e) => handleBrandChange("brandName", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="Enter your brand name"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{brandData.brandName}</p>
                    </div>
                  )}
                </div>

                {/* Industry */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Palette className="w-4 h-4 text-gray-500" />
                    Industry
                  </Label>
                  {isEditingBrand ? (
                    <Input
                      value={editedBrandData.industry}
                      onChange={(e) => handleBrandChange("industry", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="e.g., Technology, Healthcare, Finance"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{brandData.industry}</p>
                    </div>
                  )}
                </div>

                {/* Brand Description */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <FileText className="w-4 h-4 text-gray-500" />
                    Brand Description
                  </Label>
                  {isEditingBrand ? (
                    <textarea
                      value={editedBrandData.description}
                      onChange={(e) => handleBrandChange("description", e.target.value)}
                      className="w-full p-4 border-2 border-gray-300 focus:border-black rounded-lg min-h-[120px] resize-none text-gray-900"
                      placeholder="Describe your brand, its mission, and what makes it unique..."
                    />
                  ) : (
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 leading-relaxed">{brandData.description}</p>
                    </div>
                  )}
                </div>

                {/* Target Audience */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <User className="w-4 h-4 text-gray-500" />
                    Target Audience
                  </Label>
                  {isEditingBrand ? (
                    <Input
                      value={editedBrandData.targetAudience}
                      onChange={(e) => handleBrandChange("targetAudience", e.target.value)}
                      className="border-2 border-gray-300 focus:border-black h-11"
                      placeholder="Who is your primary audience?"
                    />
                  ) : (
                    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <p className="text-gray-900 font-medium">{brandData.targetAudience}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Target Market Section */}
            <div>
              <h4 className="text-lg font-semibold text-gray-900 mb-4 pb-2 border-b border-gray-200">
                Target Market
              </h4>
              <div className="space-y-6">
                {/* Target Industries */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <TrendingUp className="w-4 h-4 text-gray-500" />
                    Relevant Target Industries
                    <span className="text-xs text-gray-500 font-normal">(Max 5)</span>
                  </Label>
                  
                  {isEditingBrand && (
                    <div className="mb-3">
                      <select
                        onChange={(e) => {
                          if (e.target.value) {
                            addIndustry(e.target.value);
                            e.target.value = "";
                          }
                        }}
                        disabled={editedBrandData.targetIndustries.length >= 5}
                        className="w-full p-2.5 border-2 border-gray-300 rounded-lg focus:border-black disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
                      >
                        <option value="">
                          {editedBrandData.targetIndustries.length >= 5
                            ? "Max 5 industries selected"
                            : "Select an industry to add..."}
                        </option>
                        {commonIndustries
                          .filter((ind) => !editedBrandData.targetIndustries.includes(ind))
                          .map((industry) => (
                            <option key={industry} value={industry}>
                              {industry}
                            </option>
                          ))}
                      </select>
                    </div>
                  )}

                  <div className="p-4 bg-gray-50 rounded-lg min-h-[70px]">
                    {(isEditingBrand ? editedBrandData : brandData).targetIndustries.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {(isEditingBrand ? editedBrandData : brandData).targetIndustries.map((industry, index) => (
                          <div
                            key={index}
                            className="flex items-center gap-2 bg-gray-900 text-white px-3 py-1.5 rounded-full text-xs font-medium"
                          >
                            <span>{industry}</span>
                            {isEditingBrand && (
                              <button
                                onClick={() => removeIndustry(index)}
                                className="hover:text-red-300 transition-colors"
                              >
                                <X className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-gray-400 italic text-sm">No industries selected</p>
                    )}
                  </div>
                </div>

                {/* Target Geography */}
                <div>
                  <Label className="text-gray-700 font-medium flex items-center gap-2 mb-2 text-sm">
                    <Globe className="w-4 h-4 text-gray-500" />
                    Relevant Geography
                    <span className="text-xs text-gray-500 font-normal">(Max 5)</span>
                  </Label>
                  
                  {isEditingBrand && (
                    <div className="mb-3">
                      <select
                        onChange={(e) => {
                          if (e.target.value) {
                            addGeography(e.target.value);
                            e.target.value = "";
                          }
                        }}
                        disabled={editedBrandData.targetGeography.length >= 5}
                        className="w-full p-2.5 border-2 border-gray-300 rounded-lg focus:border-black disabled:bg-gray-100 disabled:cursor-not-allowed text-sm"
                      >
                        <option value="">
                          {editedBrandData.targetGeography.length >= 5
                            ? "Max 5 locations selected"
                            : "Select a location to add..."}
                        </option>
                        {geographyOptions
                          .filter((geo) => !editedBrandData.targetGeography.includes(geo))
                          .map((geography) => (
                            <option key={geography} value={geography}>
                              {geography}
                            </option>
                          ))}
                      </select>
                    </div>
                  )}

                  <div className="p-4 bg-gray-50 rounded-lg min-h-[70px]">
                    {(isEditingBrand ? editedBrandData : brandData).targetGeography.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {(isEditingBrand ? editedBrandData : brandData).targetGeography.map((geography, index) => (
                          <div
                            key={index}
                            className="flex items-center gap-2 bg-gray-900 text-white px-3 py-1.5 rounded-full text-xs font-medium"
                          >
                            <span>{geography}</span>
                            {isEditingBrand && (
                              <button
                                onClick={() => removeGeography(index)}
                                className="hover:text-red-300 transition-colors"
                              >
                                <X className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-gray-400 italic text-sm">No locations selected</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Right Column - Files, Preferences & Settings */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-8"
        >
          {/* File Management Card */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-8">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  Uploaded Files
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  Manage your brand assets and documents
                </p>
              </div>
              <Button onClick={triggerFileUpload} className="!bg-black hover:bg-gray-800 text-white">
                <Upload className="w-4 h-4 mr-2" />
                Upload New
              </Button>
            </div>

            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              accept="image/*,.pdf,.zip,.doc,.docx,.ppt,.pptx"
            />

            {uploadedFiles.length > 0 ? (
              <div className="space-y-3">
                {uploadedFiles.map((file) => (
                  <div
                    key={file.id}
                    className="group flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-all border border-transparent hover:border-gray-200"
                  >
                    <div className="flex items-center gap-4 flex-1">
                      <div className="w-12 h-12 bg-gradient-to-br from-gray-900 to-gray-700 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm">
                        <FileText className="w-6 h-6 text-black" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-gray-900 truncate">{file.name}</p>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs font-medium text-gray-600 bg-white px-2 py-0.5 rounded">
                            {file.type}
                          </span>
                          <span className="text-xs text-gray-500">•</span>
                          <span className="text-xs text-gray-600">{file.size}</span>
                          <span className="text-xs text-gray-500">•</span>
                          <span className="text-xs text-gray-600">{file.date}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-2 border-gray-300 hover:bg-white hover:border-gray-400"
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleFileDelete(file.id)}
                        className="border-2 border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-16 bg-gray-50 rounded-xl border-2 border-dashed border-gray-300">
                <div className="w-20 h-20 mx-auto bg-gray-100 rounded-full flex items-center justify-center mb-4">
                  <Upload className="w-10 h-10 text-gray-400" />
                </div>
                <h4 className="text-lg font-semibold text-gray-900 mb-2">No files uploaded yet</h4>
                <p className="text-sm text-gray-600 mb-6">Upload your brand assets, logos, and documents</p>
                <Button onClick={triggerFileUpload} className="!bg-black hover:bg-gray-800 text-white">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Your First File
                </Button>
              </div>
            )}
          </div>


          {/* Danger Zone */}
          <div className="bg-white rounded-2xl shadow-lg border-2 border-red-200 p-8 mb-8">
            <h3 className="text-2xl font-bold text-red-600 mb-4">Danger Zone</h3>
            <p className="text-gray-600 mb-6">
              Once you delete your account, there is no going back. Please be certain.
            </p>
            <Button
              variant="outline"
              className="border-2 border-red-500 text-red-600 hover:bg-red-50"
            >
              Delete Account
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
