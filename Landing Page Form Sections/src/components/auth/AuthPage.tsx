import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import {
  ArrowLeft,
  Mail,
  User,
  AlertCircle,
  CheckCircle2,
  Lock,
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
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Subtle background pattern */}
      <div
        className="absolute inset-0 opacity-[0.4]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239ca3af' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v6h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }}
      />

      <div className="relative z-10">
        {/* Header */}
        <header className="bg-white/95 backdrop-blur-xl border-b border-gray-200 shadow-sm">
          <div
            className="max-w-7xl mx-auto px-8"
            style={{ paddingTop: "1.375rem", paddingBottom: "1.375rem" }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={onBackToLanding}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-all duration-200 mr-1"
                  title="Back to landing page"
                >
                  <ArrowLeft className="w-5 h-5 text-gray-700" />
                </button>
                <img
                  src="/1syx-logo.jpeg"
                  alt="1SYX Logo"
                  className="h-10 w-auto"
                />
                <span className="text-2xl font-bold text-gray-900 tracking-tight">
                  1SYX
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Auth Form */}
        <div className="flex items-center justify-center min-h-[calc(100vh-88px)] py-12 px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="w-full"
            style={{ maxWidth: "440px" }}
          >
            <div className="bg-white rounded-2xl shadow-2xl p-8 border border-gray-200">
              {/* Header */}
              <div className="text-center mb-6">
                <motion.div
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 200 }}
                  className="w-14 h-14 bg-black rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg p-2"
                >
                  <img
                    src="/1syx-logo.jpeg"
                    alt="1SYX Logo"
                    className="w-full h-full object-contain rounded-lg"
                  />
                </motion.div>
                <h2 className="text-2xl font-bold mb-2 text-gray-900">
                  {isLogin ? "Welcome Back" : "Create Account"}
                </h2>
                <p className="text-gray-600 text-sm">
                  {isLogin
                    ? "Sign in to continue to 1SYX"
                    : "Sign up to get started with 1SYX"}
                </p>
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {errors.general && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="bg-red-50 border border-red-200 rounded-xl p-3 mb-5 flex items-center gap-3"
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
                        className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-sm"
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
                        className={`border-2 rounded-xl transition-all duration-200 h-11 ${
                          errors.name
                            ? "border-red-300 focus:border-red-500"
                            : "border-gray-300 focus:border-black"
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
                    className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-sm"
                  >
                    <Mail className="w-4 h-4" />
                    {isLogin ? "Email" : "Work Email"}
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => handleInputChange("email", e.target.value)}
                    className={`border-2 rounded-xl transition-all duration-200 h-11 ${
                      errors.email
                        ? "border-red-300 focus:border-red-500"
                        : "border-gray-300 focus:border-black"
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
                    className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-sm"
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
                    className={`border-2 rounded-xl transition-all duration-200 h-11 ${
                      errors.password
                        ? "border-red-300 focus:border-red-500"
                        : "border-gray-300 focus:border-black"
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
                        className="text-gray-900 font-semibold flex items-center gap-2 mb-2 text-sm"
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
                        className={`border-2 rounded-xl transition-all duration-200 h-11 ${
                          errors.confirmPassword
                            ? "border-red-300 focus:border-red-500"
                            : "border-gray-300 focus:border-black"
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
                  className="pt-1"
                >
                  <Button
                    type="submit"
                    disabled={isLoading}
                    className="w-full !bg-black hover:!bg-gray-800 !text-white !border-0 shadow-lg rounded-xl h-11 font-semibold text-sm transition-all duration-200 disabled:opacity-50"
                  >
                    <span className="flex items-center justify-center gap-2">
                      {isLoading ? (
                        "Processing..."
                      ) : (
                        <>
                          {isLogin ? "Sign In" : "Create Account"}
                          <CheckCircle2 className="w-4 h-4" />
                        </>
                      )}
                    </span>
                  </Button>
                </motion.div>
              </form>

              {/* Toggle Mode */}
              <div className="mt-5 text-center">
                <p className="text-gray-600 text-sm">
                  {isLogin
                    ? "Don't have an account?"
                    : "Already have an account?"}{" "}
                  <button
                    type="button"
                    onClick={toggleMode}
                    className="text-black hover:text-gray-700 font-semibold transition-colors underline decoration-2 underline-offset-2"
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
