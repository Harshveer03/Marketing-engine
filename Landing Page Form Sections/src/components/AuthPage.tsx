import { useState, FormEvent } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  ArrowLeft,
  Mail,
  Lock,
  User,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

interface AuthPageProps {
  onAuthSuccess: () => void;
  onBackToLanding: () => void;
}

// Dummy account for testing
const DUMMY_ACCOUNT = {
  email: "test@company.com",
  password: "Harshveer@24",
  name: "Test User",
};

// Personal email domains to block
const PERSONAL_EMAIL_DOMAINS = [
  "gmail.com",
  "yahoo.com",
  "hotmail.com",
  "outlook.com",
  "aol.com",
  "icloud.com",
  "mail.com",
  "protonmail.com",
  "yandex.com",
  "zoho.com",
];

export function AuthPage({ onAuthSuccess, onBackToLanding }: AuthPageProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);

  const validateWorkEmail = (email: string): boolean => {
    const domain = email.split("@")[1]?.toLowerCase();
    return !PERSONAL_EMAIL_DOMAINS.includes(domain);
  };

  const validatePassword = (
    password: string
  ): { valid: boolean; message?: string } => {
    if (password.length < 8) {
      return {
        valid: false,
        message: "Password must be at least 8 characters",
      };
    }
    if (!/[A-Z]/.test(password)) {
      return {
        valid: false,
        message: "Password must contain at least 1 uppercase letter",
      };
    }
    return { valid: true };
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: "" }));
    }
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    // Validate email
    if (!formData.email) {
      newErrors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Invalid email format";
    } else if (!isLogin && !validateWorkEmail(formData.email)) {
      newErrors.email =
        "Please use a work email address (personal emails not allowed)";
    }

    // Validate password
    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (!isLogin) {
      const passwordValidation = validatePassword(formData.password);
      if (!passwordValidation.valid) {
        newErrors.password = passwordValidation.message!;
      }
    }

    // Validate name for signup
    if (!isLogin && !formData.name.trim()) {
      newErrors.name = "Name is required";
    }

    // Validate confirm password for signup
    if (!isLogin && formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    // Simulate authentication
    setIsLoading(true);
    setTimeout(() => {
      if (isLogin) {
        // Login validation
        if (
          formData.email === DUMMY_ACCOUNT.email &&
          formData.password === DUMMY_ACCOUNT.password
        ) {
          localStorage.setItem("isAuthenticated", "true");
          localStorage.setItem("userEmail", formData.email);
          localStorage.setItem("userName", DUMMY_ACCOUNT.name);
          onAuthSuccess();
        } else {
          setErrors({ general: "Invalid email or password" });
        }
      } else {
        // Sign up - store new user
        localStorage.setItem("isAuthenticated", "true");
        localStorage.setItem("userEmail", formData.email);
        localStorage.setItem("userName", formData.name);
        onAuthSuccess();
      }
      setIsLoading(false);
    }, 1000);
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setFormData({ name: "", email: "", password: "", confirmPassword: "" });
    setErrors({});
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden">
      {/* Background Pattern */}
      <div
        className="absolute inset-0 opacity-20 bg-cover bg-center"
        style={{
          backgroundImage: `url('https://images.unsplash.com/photo-1557682250-33bd709cbe85?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwdXJwbGUlMjBibHVlJTIwZ3JhZGllbnR8ZW58MXx8fHwxNzYzMDA0MzMwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
        }}
      />

      <div className="relative z-10">
        {/* Header */}
        <header className="bg-white/95 backdrop-blur-md border-b border-indigo-100 shadow-sm">
          <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                onClick={onBackToLanding}
                className="p-2 hover:bg-indigo-50 rounded-lg transition-all duration-200 mr-2"
                title="Back to landing page"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold">BG</span>
              </div>
              <span className="text-xl font-semibold text-gray-800">
                BrandGen
              </span>
            </div>
          </div>
        </header>

        {/* Auth Form */}
        <div className="flex items-center justify-center min-h-[calc(100vh-80px)] p-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="w-full"
            style={{ maxWidth: "420px" }}
          >
            <div className="bg-white rounded-2xl shadow-xl p-8 border border-indigo-100">
              {/* Header */}
              <div className="text-center mb-6">
                <motion.div
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 200 }}
                  className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center mx-auto mb-3"
                >
                  <Lock className="w-7 h-7 text-white" />
                </motion.div>
                <h2 className="text-2xl mb-1">
                  {isLogin ? "Welcome Back" : "Create Account"}
                </h2>
                <p className="text-gray-600 text-sm">
                  {isLogin
                    ? "Sign in to continue to BrandGen"
                    : "Sign up to get started with BrandGen"}
                </p>
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {errors.general && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="bg-red-50 border border-red-200 rounded-xl p-3 mb-4 flex items-center gap-3"
                  >
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    <p className="text-red-700 text-sm">{errors.general}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Name Field (Sign Up Only) */}
                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Label
                        htmlFor="name"
                        className="text-gray-700 flex items-center gap-2"
                      >
                        <User className="w-4 h-4" />
                        Full Name
                      </Label>
                      <Input
                        id="name"
                        type="text"
                        value={formData.name}
                        onChange={(e) =>
                          handleInputChange("name", e.target.value)
                        }
                        className={`mt-2 border-2 rounded-xl transition-all duration-200 ${
                          errors.name
                            ? "border-red-300 focus:border-red-400"
                            : "border-indigo-200 focus:border-indigo-400"
                        }`}
                        placeholder="John Doe"
                      />
                      {errors.name && (
                        <p className="text-red-500 text-sm mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.name}
                        </p>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Email Field */}
                <div>
                  <Label
                    htmlFor="email"
                    className="text-gray-700 flex items-center gap-2"
                  >
                    <Mail className="w-4 h-4" />
                    {isLogin ? "Email" : "Work Email"}
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => handleInputChange("email", e.target.value)}
                    className={`mt-2 border-2 rounded-xl transition-all duration-200 ${
                      errors.email
                        ? "border-red-300 focus:border-red-400"
                        : "border-indigo-200 focus:border-indigo-400"
                    }`}
                    placeholder={
                      isLogin ? "your@email.com" : "your@company.com"
                    }
                  />
                  {errors.email && (
                    <p className="text-red-500 text-sm mt-1 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />
                      {errors.email}
                    </p>
                  )}
                  {!isLogin && !errors.email && (
                    <p className="text-gray-500 text-xs mt-1">
                      Personal emails (@gmail.com, @yahoo.com, etc.) are not
                      allowed
                    </p>
                  )}
                </div>

                {/* Password Field */}
                <div>
                  <Label
                    htmlFor="password"
                    className="text-gray-700 flex items-center gap-2"
                  >
                    <Lock className="w-4 h-4" />
                    Password
                  </Label>
                  <Input
                    id="password"
                    type="password"
                    value={formData.password}
                    onChange={(e) =>
                      handleInputChange("password", e.target.value)
                    }
                    className={`mt-2 border-2 rounded-xl transition-all duration-200 ${
                      errors.password
                        ? "border-red-300 focus:border-red-400"
                        : "border-indigo-200 focus:border-indigo-400"
                    }`}
                    placeholder="••••••••"
                  />
                  {errors.password && (
                    <p className="text-red-500 text-sm mt-1 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" />
                      {errors.password}
                    </p>
                  )}
                  {!isLogin && !errors.password && (
                    <p className="text-gray-500 text-xs mt-1">
                      Min 8 characters, 1 uppercase letter, special characters
                      allowed
                    </p>
                  )}
                </div>

                {/* Confirm Password Field (Sign Up Only) */}
                <AnimatePresence>
                  {!isLogin && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Label
                        htmlFor="confirmPassword"
                        className="text-gray-700 flex items-center gap-2"
                      >
                        <Lock className="w-4 h-4" />
                        Confirm Password
                      </Label>
                      <Input
                        id="confirmPassword"
                        type="password"
                        value={formData.confirmPassword}
                        onChange={(e) =>
                          handleInputChange("confirmPassword", e.target.value)
                        }
                        className={`mt-2 border-2 rounded-xl transition-all duration-200 ${
                          errors.confirmPassword
                            ? "border-red-300 focus:border-red-400"
                            : "border-indigo-200 focus:border-indigo-400"
                        }`}
                        placeholder="••••••••"
                      />
                      {errors.confirmPassword && (
                        <p className="text-red-500 text-sm mt-1 flex items-center gap-1">
                          <AlertCircle className="w-3 h-3" />
                          {errors.confirmPassword}
                        </p>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Submit Button */}
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white border-0 shadow-lg rounded-xl py-3 relative overflow-hidden"
                  >
                    <div
                      className="absolute inset-0 opacity-20 bg-cover bg-center"
                      style={{
                        backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                      }}
                    />
                    <span className="relative z-10 flex items-center justify-center gap-2">
                      {isLoading ? (
                        "Processing..."
                      ) : (
                        <>
                          {isLogin ? "Sign In" : "Create Account"}
                          <CheckCircle2 className="w-5 h-5" />
                        </>
                      )}
                    </span>
                  </Button>
                </motion.div>
              </form>

              {/* Toggle Mode */}
              <div className="mt-4 text-center">
                <p className="text-gray-600 text-sm">
                  {isLogin
                    ? "Don't have an account?"
                    : "Already have an account?"}{" "}
                  <button
                    type="button"
                    onClick={toggleMode}
                    className="text-indigo-600 hover:text-indigo-700 font-semibold transition-colors"
                  >
                    {isLogin ? "Sign Up" : "Sign In"}
                  </button>
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
