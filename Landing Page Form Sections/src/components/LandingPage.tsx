import { Button } from "./ui/button";
import {
  ArrowRight,
  CheckCircle,
  Zap,
  Target,
  Sparkles,
  BarChart,
  Users,
  Clock,
  ChevronDown,
} from "lucide-react";
import { motion } from "motion/react";
import { ImageWithFallback } from "./figma/ImageWithFallback";

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  const features = [
    {
      icon: Zap,
      title: "AI-Powered Insights",
      description:
        "Get intelligent brand recommendations powered by advanced AI technology.",
    },
    {
      icon: Target,
      title: "Targeted Strategy",
      description:
        "Create brand strategies that resonate with your specific audience.",
    },
    {
      icon: Sparkles,
      title: "Creative Excellence",
      description:
        "Generate stunning brand materials that stand out from the competition.",
    },
    {
      icon: BarChart,
      title: "Performance Analytics",
      description:
        "Track and measure your brand performance with detailed analytics.",
    },
    {
      icon: Users,
      title: "Team Collaboration",
      description:
        "Work seamlessly with your team on brand development projects.",
    },
    {
      icon: Clock,
      title: "Fast Turnaround",
      description: "Get your brand materials ready in minutes, not weeks.",
    },
  ];

  const steps = [
    {
      number: "01",
      title: "Define Your Goals",
      description: "Tell us what you want to achieve with your brand.",
    },
    {
      number: "02",
      title: "Share Your Info",
      description: "Provide details about your brand, industry, and audience.",
    },
    {
      number: "03",
      title: "Get Your Score",
      description:
        "Receive a comprehensive brand analysis and recommendations.",
    },
    {
      number: "04",
      title: "Generate Assets",
      description: "Create professional brand materials instantly.",
    },
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "CEO, TechStart Inc",
      content:
        "BrandGen transformed our brand strategy in just days. The AI insights were spot-on and helped us refine our messaging perfectly.",
      image:
        "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&h=150&fit=crop",
    },
    {
      name: "Michael Chen",
      role: "Marketing Director, Fusion Co",
      content:
        "The speed and quality of brand materials we got was incredible. This tool saved us thousands in agency fees.",
      image:
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop",
    },
    {
      name: "Emily Rodriguez",
      role: "Founder, Creative Studio",
      content:
        "As a designer, I was skeptical at first, but BrandGen exceeded my expectations. It's now an essential part of our workflow.",
      image:
        "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop",
    },
  ];

  const faqs = [
    {
      question: "How does BrandGen work?",
      answer:
        "BrandGen uses advanced AI to analyze your brand inputs and generate customized recommendations, strategies, and materials tailored to your specific needs.",
    },
    {
      question: "What kind of results can I expect?",
      answer:
        "You'll receive a comprehensive brand score, strategic recommendations, and ready-to-use brand materials including logos, color palettes, and marketing content.",
    },
    {
      question: "Is my data secure?",
      answer:
        "Absolutely. We use industry-standard encryption and security measures to protect your brand information and intellectual property.",
    },
    {
      question: "Can I collaborate with my team?",
      answer:
        "Yes! BrandGen supports team collaboration features, allowing multiple stakeholders to contribute and review brand development.",
    },
  ];

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
        {/* Navigation */}
        <nav className="bg-white/95 backdrop-blur-md border-b border-indigo-100 shadow-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold">BG</span>
              </div>
              <span className="text-xl font-semibold text-gray-800">
                BrandGen
              </span>
            </div>

            <div className="flex items-center gap-8">
              <a
                href="#features"
                className="text-gray-600 hover:text-indigo-600 transition-colors"
              >
                Features
              </a>
              <a
                href="#how-it-works"
                className="text-gray-600 hover:text-indigo-600 transition-colors"
              >
                How It Works
              </a>
              <a
                href="#testimonials"
                className="text-gray-600 hover:text-indigo-600 transition-colors"
              >
                Testimonials
              </a>
              <a
                href="#faq"
                className="text-gray-600 hover:text-indigo-600 transition-colors"
              >
                FAQ
              </a>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  variant="outline"
                  className="border-2 border-indigo-300 text-indigo-700 hover:bg-indigo-50 rounded-lg"
                >
                  Login
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white border-0 shadow-md rounded-lg"
                >
                  Get Started
                </Button>
              </motion.div>
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="max-w-7xl mx-auto px-8 py-20">
          <div className="grid grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="inline-block bg-indigo-100 text-indigo-700 px-4 py-2 rounded-full mb-6"
              >
                ✨ AI-Powered Brand Intelligence
              </motion.div>
              <h1 className="text-6xl mb-6 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Build Your Brand with AI Precision
              </h1>
              <p className="text-xl text-gray-600 mb-8">
                Transform your brand vision into reality with intelligent
                insights, strategic recommendations, and stunning creative
                assets—all in one powerful platform.
              </p>
              <div className="flex gap-4">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    onClick={onGetStarted}
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white border-0 shadow-xl rounded-xl px-8 py-6 text-lg relative overflow-hidden"
                  >
                    <div
                      className="absolute inset-0 opacity-30 bg-cover bg-center"
                      style={{
                        backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                      }}
                    />
                    <span className="relative z-10 flex items-center gap-2">
                      Start Building Your Brand
                      <ArrowRight className="w-5 h-5" />
                    </span>
                  </Button>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="outline"
                    className="border-2 border-indigo-300 text-indigo-700 hover:bg-indigo-50 rounded-xl px-8 py-6 text-lg"
                  >
                    Watch Demo
                  </Button>
                </motion.div>
              </div>

              <div className="flex items-center gap-8 mt-12">
                <div>
                  <div className="text-3xl font-bold text-indigo-600">10K+</div>
                  <div className="text-gray-600">Brands Created</div>
                </div>
                <div className="w-px h-12 bg-indigo-200"></div>
                <div>
                  <div className="text-3xl font-bold text-indigo-600">98%</div>
                  <div className="text-gray-600">Satisfaction Rate</div>
                </div>
                <div className="w-px h-12 bg-indigo-200"></div>
                <div>
                  <div className="text-3xl font-bold text-indigo-600">24/7</div>
                  <div className="text-gray-600">AI Support</div>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="relative"
            >
              <div className="bg-white rounded-3xl shadow-2xl p-8 relative overflow-hidden">
                <div
                  className="absolute inset-0 opacity-10 bg-cover bg-center"
                  style={{
                    backgroundImage: `url('https://images.unsplash.com/photo-1670225597315-782633cfbd2a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtaW5pbWFsJTIwZ2VvbWV0cmljJTIwcGF0dGVybnxlbnwxfHx8fDE3NjMwMDUxMjF8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
                  }}
                />
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=600&fit=crop"
                  alt="Brand Dashboard"
                  className="rounded-2xl shadow-lg w-full relative z-10"
                />
              </div>
              <div className="absolute -top-6 -right-6 w-24 h-24 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full blur-2xl opacity-50"></div>
              <div className="absolute -bottom-6 -left-6 w-32 h-32 bg-gradient-to-br from-blue-400 to-indigo-500 rounded-full blur-2xl opacity-50"></div>
            </motion.div>
          </div>
        </section>

        {/* Trusted By Section */}
        <section className="bg-white/80 backdrop-blur-sm py-12 border-y border-indigo-100">
          <div className="max-w-7xl mx-auto px-8">
            <p className="text-center text-gray-600 mb-8">
              Trusted by leading brands worldwide
            </p>
            <div className="flex justify-center items-center gap-12 opacity-60">
              <div className="text-2xl font-bold text-gray-700">ACME Corp</div>
              <div className="text-2xl font-bold text-gray-700">TechVision</div>
              <div className="text-2xl font-bold text-gray-700">Innovate</div>
              <div className="text-2xl font-bold text-gray-700">
                BrightFuture
              </div>
              <div className="text-2xl font-bold text-gray-700">NextGen</div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="max-w-7xl mx-auto px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl mb-4">
              Powerful Features for Brand Success
            </h2>
            <p className="text-xl text-gray-600">
              Everything you need to build, refine, and scale your brand
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                whileHover={{ y: -8 }}
                className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-indigo-100"
              >
                <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center mb-6">
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl mb-3 text-gray-800">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* How It Works Section */}
        <section
          id="how-it-works"
          className="bg-gradient-to-br from-indigo-500 to-purple-500 py-20 relative overflow-hidden"
        >
          <div
            className="absolute inset-0 opacity-20 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
            }}
          />
          <div className="max-w-7xl mx-auto px-8 relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 text-white">How It Works</h2>
              <p className="text-xl text-indigo-100">
                Get your brand ready in 4 simple steps
              </p>
            </motion.div>

            <div className="grid grid-cols-4 gap-8">
              {steps.map((step, index) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="relative"
                >
                  <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 hover:bg-white/20 transition-all duration-300">
                    <div className="text-5xl font-bold text-white/30 mb-4">
                      {step.number}
                    </div>
                    <h3 className="text-xl mb-3 text-white">{step.title}</h3>
                    <p className="text-indigo-100">{step.description}</p>
                  </div>
                  {index < steps.length - 1 && (
                    <div className="hidden xl:block absolute top-1/2 -right-4 transform -translate-y-1/2">
                      <ArrowRight className="w-8 h-8 text-white/40" />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.6 }}
              className="text-center mt-12"
            >
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  className="bg-white text-indigo-600 hover:bg-gray-50 border-0 shadow-xl rounded-xl px-8 py-6 text-lg"
                >
                  Start Your Brand Journey
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section id="testimonials" className="max-w-7xl mx-auto px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl mb-4">What Our Clients Say</h2>
            <p className="text-xl text-gray-600">
              Join thousands of satisfied brand builders
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={testimonial.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                className="bg-white rounded-2xl p-8 shadow-lg border border-indigo-100"
              >
                <div className="flex items-center gap-4 mb-6">
                  <ImageWithFallback
                    src={testimonial.image}
                    alt={testimonial.name}
                    className="w-16 h-16 rounded-full object-cover"
                  />
                  <div>
                    <div className="font-semibold text-gray-800">
                      {testimonial.name}
                    </div>
                    <div className="text-sm text-gray-600">
                      {testimonial.role}
                    </div>
                  </div>
                </div>
                <p className="text-gray-600 italic">"{testimonial.content}"</p>
                <div className="flex gap-1 mt-4">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className="text-yellow-400">
                      ★
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* FAQ Section */}
        <section
          id="faq"
          className="bg-white/80 backdrop-blur-sm py-20 border-y border-indigo-100"
        >
          <div className="max-w-4xl mx-auto px-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4">Frequently Asked Questions</h2>
              <p className="text-xl text-gray-600">
                Everything you need to know about BrandGen
              </p>
            </motion.div>

            <div className="space-y-6">
              {faqs.map((faq, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="bg-white rounded-2xl p-8 shadow-lg border border-indigo-100"
                >
                  <h3 className="text-xl mb-3 text-gray-800 flex items-center gap-3">
                    <CheckCircle className="w-6 h-6 text-indigo-600" />
                    {faq.question}
                  </h3>
                  <p className="text-gray-600 ml-9">{faq.answer}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="max-w-7xl mx-auto px-8 py-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="bg-gradient-to-r from-indigo-500 to-purple-500 rounded-3xl p-16 text-center relative overflow-hidden shadow-2xl"
          >
            <div
              className="absolute inset-0 opacity-20 bg-cover bg-center"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
              }}
            />
            <div className="relative z-10">
              <h2 className="text-5xl mb-6 text-white">
                Ready to Transform Your Brand?
              </h2>
              <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
                Join thousands of businesses using BrandGen to create powerful,
                memorable brands that drive results.
              </p>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onGetStarted}
                  className="bg-white text-indigo-600 hover:bg-gray-50 border-0 shadow-xl rounded-xl px-10 py-6 text-lg"
                >
                  Get Started for Free
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </motion.div>
              <p className="text-indigo-100 mt-4">
                No credit card required • 14-day free trial
              </p>
            </div>
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="bg-gray-900 text-gray-300 py-12">
          <div className="max-w-7xl mx-auto px-8">
            <div className="grid grid-cols-4 gap-8 mb-8">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">BG</span>
                  </div>
                  <span className="text-xl font-semibold text-white">
                    BrandGen
                  </span>
                </div>
                <p className="text-gray-400">
                  Building brands with AI precision.
                </p>
              </div>

              <div>
                <h4 className="text-white mb-4">Product</h4>
                <ul className="space-y-2">
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Features
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Pricing
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Updates
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Company</h4>
                <ul className="space-y-2">
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      About
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Blog
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Careers
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Legal</h4>
                <ul className="space-y-2">
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Terms
                    </a>
                  </li>
                  <li>
                    <a
                      href="#"
                      className="hover:text-indigo-400 transition-colors"
                    >
                      Contact
                    </a>
                  </li>
                </ul>
              </div>
            </div>

            <div className="border-t border-gray-800 pt-8 flex justify-between items-center">
              <p className="text-gray-500">
                © 2024 BrandGen. All rights reserved.
              </p>
              <div className="flex gap-4">
                <a
                  href="#"
                  className="text-gray-400 hover:text-indigo-400 transition-colors"
                >
                  Twitter
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-indigo-400 transition-colors"
                >
                  LinkedIn
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-indigo-400 transition-colors"
                >
                  Instagram
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
