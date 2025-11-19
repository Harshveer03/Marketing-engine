import { Button } from "../ui/button";
import {
  ArrowRight,
  CheckCircle,
  Zap,
  Target,
  Sparkles,
  BarChart,
  Users,
  Clock,
  Plus,
  Minus,
  HelpCircle,
  Shield,
  DollarSign,
  Settings,
  ThumbsUp,
  ThumbsDown,
  MessageCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { ImageWithFallback } from "../figma/ImageWithFallback";
import { useState } from "react";

interface LandingPageProps {
  onGetStarted: () => void;
}

export function LandingPage({ onGetStarted }: LandingPageProps) {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(null);
  const [helpfulVotes, setHelpfulVotes] = useState<{
    [key: number]: "yes" | "no" | null;
  }>({});
  const features = [
    {
      icon: Zap,
      title: "AI-Powered Insights",
      description:
        "Get intelligent brand recommendations powered by advanced AI technology.",
      bgImage:
        "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=600&fit=crop",
    },
    {
      icon: Target,
      title: "Targeted Strategy",
      description:
        "Create brand strategies that resonate with your specific audience.",
      bgImage:
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop",
    },
    {
      icon: Sparkles,
      title: "Creative Excellence",
      description:
        "Generate stunning brand materials that stand out from the competition.",
      bgImage:
        "https://images.unsplash.com/photo-1561070791-2526d30994b5?w=800&h=600&fit=crop",
    },
    {
      icon: BarChart,
      title: "Performance Analytics",
      description:
        "Track and measure your brand performance with detailed analytics.",
      bgImage:
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=600&fit=crop",
    },
    {
      icon: Users,
      title: "Team Collaboration",
      description:
        "Work seamlessly with your team on brand development projects.",
      bgImage:
        "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=800&h=600&fit=crop",
    },
    {
      icon: Clock,
      title: "Fast Turnaround",
      description: "Get your brand materials ready in minutes, not weeks.",
      bgImage:
        "https://images.unsplash.com/photo-1501139083538-0139583c060f?w=800&h=600&fit=crop",
    },
  ];

  const steps = [
    {
      number: "01",
      title: "Share Your Info",
      description: "Provide details about your brand, industry, and audience.",
    },
    {
      number: "02",
      title: "Get Your Score",
      description:
        "Receive a comprehensive brand analysis and recommendations.",
    },
    {
      number: "03",
      title: "Define Your Goals",
      description: "Tell us what you want to achieve with your brand.",
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
        "1SYX transformed our brand strategy in just days. The AI insights were spot-on and helped us refine our messaging perfectly.",
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
        "As a designer, I was skeptical at first, but 1SYX exceeded my expectations. It's now an essential part of our workflow.",
      image:
        "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop",
    },
  ];

  const faqs = [
    {
      icon: HelpCircle,
      question: "How does 1SYX work?",
      answer:
        "1SYX uses advanced AI to analyze your brand inputs and generate customized recommendations, strategies, and materials tailored to your specific needs. Simply share your brand information, and our AI will provide comprehensive insights and assets.",
      category: "Getting Started",
    },
    {
      icon: Clock,
      question: "How long does it take to get results?",
      answer:
        "You'll receive your brand score and initial recommendations within minutes. Complete brand materials and assets are typically ready within 24 hours, depending on the complexity of your requirements.",
      category: "Getting Started",
    },
    {
      icon: Target,
      question: "What kind of results can I expect?",
      answer:
        "You'll receive a comprehensive brand score, strategic recommendations, and ready-to-use brand materials including logos, color palettes, typography guidelines, and marketing content tailored to your industry and audience.",
      category: "Features",
    },
    {
      icon: Users,
      question: "Can I collaborate with my team?",
      answer:
        "Yes! 1SYX supports team collaboration features, allowing multiple stakeholders to contribute and review brand development. You can invite team members, share feedback, and work together seamlessly.",
      category: "Features",
    },
    {
      icon: DollarSign,
      question: "What are the pricing options?",
      answer:
        "We offer flexible pricing plans to suit businesses of all sizes. Start with our free trial to explore the platform, then choose from monthly or annual subscriptions. Enterprise plans are available for larger organizations with custom needs.",
      category: "Pricing",
    },
    {
      icon: Shield,
      question: "Is my data secure?",
      answer:
        "Absolutely. We use industry-standard encryption and security measures to protect your brand information and intellectual property. Your data is stored securely and never shared with third parties without your explicit consent.",
      category: "Security",
    },
  ];

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Grid pattern background */}
      <div
        className="absolute inset-0 opacity-[0.15]"
        style={{
          backgroundImage: `linear-gradient(to right, #d1d5db 1px, transparent 1px), linear-gradient(to bottom, #d1d5db 1px, transparent 1px)`,
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10">
        {/* Navigation */}
        <nav className="bg-white/95 backdrop-blur-md border-b border-gray-200 shadow-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img
                src="/1syx-logo.jpeg"
                alt="1SYX Logo"
                className="h-10 w-auto"
              />
              <span className="text-xl font-bold text-gray-900 tracking-wider">
                1SYX
              </span>
            </div>

            <div className="flex items-center gap-8">
              <a
                href="#features"
                className="text-gray-600 hover:text-black transition-colors"
              >
                Features
              </a>
              <a
                href="#how-it-works"
                className="text-gray-600 hover:text-black transition-colors"
              >
                How It Works
              </a>
              <a
                href="#testimonials"
                className="text-gray-600 hover:text-black transition-colors"
              >
                Testimonials
              </a>
              <a
                href="#faq"
                className="text-gray-600 hover:text-black transition-colors"
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
                  className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white font-semibold"
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
                  className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-md"
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
                className="inline-block bg-gray-100 text-gray-800 px-4 py-2 rounded-full mb-6"
              >
                ✨ AI-Powered Brand Intelligence
              </motion.div>
              <h1 className="text-6xl mb-6 font-bold text-gray-900">
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
                    className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-xl rounded-xl px-8 py-6 text-lg font-semibold"
                  >
                    <span className="flex items-center gap-2">
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
                    className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white rounded-xl px-8 py-6 text-lg font-semibold"
                  >
                    Watch Demo
                  </Button>
                </motion.div>
              </div>

              <div className="flex items-center gap-8 mt-12">
                <div>
                  <div className="text-3xl font-bold text-gray-900">10K+</div>
                  <div className="text-gray-600">Brands Created</div>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <div className="text-3xl font-bold text-gray-900">98%</div>
                  <div className="text-gray-600">Satisfaction Rate</div>
                </div>
                <div className="w-px h-12 bg-gray-300"></div>
                <div>
                  <div className="text-3xl font-bold text-gray-900">24/7</div>
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
        <section className="bg-gray-900 py-12 border-y border-gray-200">
          <div className="max-w-7xl mx-auto px-8">
            <p className="text-center text-gray-600 mb-8 font-medium text-xl">
              Trusted by leading brands worldwide
            </p>
            <div className="flex justify-center items-center gap-12">
              <div className="text-2xl font-bold text-gray-400">ACME Corp</div>
              <div className="text-2xl font-bold text-gray-400">TechVision</div>
              <div className="text-2xl font-bold text-gray-400">Innovate</div>
              <div className="text-2xl font-bold text-gray-400">
                BrightFuture
              </div>
              <div className="text-2xl font-bold text-gray-400">NextGen</div>
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
                className="bg-white p-8 shadow-lg hover:shadow-2xl transition-all duration-300 border border-gray-200 relative overflow-hidden group"
              >
                {/* Background Image - Very Subtle */}
                <div
                  className="absolute inset-0 opacity-[0.15] group-hover:opacity-[0.12] transition-opacity duration-500 bg-cover bg-center"
                  style={{ backgroundImage: `url('${feature.bgImage}')` }}
                />

                {/* White overlay for text readability */}
                <div className="absolute inset-0 bg-white/50" />

                <div className="relative z-10">
                  <div className="w-14 h-14 bg-black flex items-center justify-center mb-6">
                    <feature.icon className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl mb-3 text-gray-900 font-bold">
                    {feature.title}
                  </h3>
                  <p className="text-gray-800 font-medium leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* How It Works Section */}
        <section
          id="how-it-works"
          className="bg-gray-900 py-20 relative overflow-hidden"
        >
          {/* Background Image */}
          <div
            className="absolute inset-0 opacity-10 bg-cover bg-center"
            style={{
              backgroundImage: `url('https://images.unsplash.com/photo-1557804506-669a67965ba0?w=1920&h=1080&fit=crop')`,
            }}
          />

          {/* Gradient Overlay */}
          <div className="absolute inset-0 bg-gradient-to-b from-gray-900/90 via-gray-900/80 to-gray-900/90" />

          <div className="max-w-7xl mx-auto px-8 relative z-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-5xl mb-4 text-white">How It Works</h2>
              <p className="text-xl text-gray-300">
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
                  className="relative flex"
                >
                  <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20 hover:bg-white/20 transition-all duration-300 flex flex-col h-full w-full">
                    <div className="text-5xl font-bold text-white/30 mb-4">
                      {step.number}
                    </div>
                    <h3 className="text-xl mb-3 text-white">{step.title}</h3>
                    <p className="text-gray-300 flex-grow">
                      {step.description}
                    </p>
                  </div>
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
                  className="bg-white text-black hover:bg-gray-100 border-0 shadow-xl rounded-xl px-8 py-6 text-lg"
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
                className="bg-white rounded-2xl p-8 shadow-lg border border-gray-200"
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
        <section id="faq" className="bg-white py-20 border-y border-gray-200">
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
                Everything you need to know about 1SYX
              </p>
            </motion.div>

            <div className="space-y-4">
              {faqs.map((faq, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.1 }}
                  className="bg-white rounded-2xl shadow-lg border-2 border-gray-200 overflow-hidden hover:shadow-xl hover:border-gray-300 transition-all duration-300"
                >
                  <button
                    onClick={() =>
                      setOpenFaqIndex(openFaqIndex === index ? null : index)
                    }
                    className="w-full p-6 flex items-center justify-between text-left hover:bg-gray-50 transition-colors duration-200"
                  >
                    <div className="flex items-center gap-4 flex-1">
                      {/* Question Number */}
                      <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center flex-shrink-0">
                        <span className="text-sm font-bold text-gray-700">
                          {index + 1}
                        </span>
                      </div>

                      {/* Icon */}
                      <div className="w-12 h-12 bg-black rounded-xl flex items-center justify-center flex-shrink-0">
                        <faq.icon className="w-6 h-6 text-white" />
                      </div>

                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                            {faq.category}
                          </span>
                        </div>
                        <h3 className="text-lg font-bold text-gray-900">
                          {faq.question}
                        </h3>
                      </div>
                    </div>
                    <motion.div
                      animate={{ rotate: openFaqIndex === index ? 180 : 0 }}
                      transition={{ duration: 0.3 }}
                      className="flex-shrink-0 ml-4"
                    >
                      {openFaqIndex === index ? (
                        <Minus className="w-6 h-6 text-gray-900" />
                      ) : (
                        <Plus className="w-6 h-6 text-gray-900" />
                      )}
                    </motion.div>
                  </button>

                  <AnimatePresence>
                    {openFaqIndex === index && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <div className="px-6 pb-6 pt-2 border-t border-gray-100">
                          <div className="pl-20 pr-10">
                            <p className="text-gray-700 leading-relaxed mb-6">
                              {faq.answer}
                            </p>

                            {/* Was this helpful section */}
                            <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
                              <span className="text-sm text-gray-600 font-medium">
                                Was this helpful?
                              </span>
                              <div className="flex gap-2">
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setHelpfulVotes({
                                      ...helpfulVotes,
                                      [index]: "yes",
                                    });
                                  }}
                                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${
                                    helpfulVotes[index] === "yes"
                                      ? "bg-green-50 border-green-500 text-green-700"
                                      : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                  }`}
                                >
                                  <ThumbsUp className="w-4 h-4" />
                                  <span className="text-sm font-medium">
                                    Yes
                                  </span>
                                </button>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setHelpfulVotes({
                                      ...helpfulVotes,
                                      [index]: "no",
                                    });
                                  }}
                                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all duration-200 ${
                                    helpfulVotes[index] === "no"
                                      ? "bg-red-50 border-red-500 text-red-700"
                                      : "border-gray-300 text-gray-600 hover:bg-gray-50"
                                  }`}
                                >
                                  <ThumbsDown className="w-4 h-4" />
                                  <span className="text-sm font-medium">
                                    No
                                  </span>
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </div>

            {/* Still have questions CTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
              className="mt-12 text-center"
            >
              <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-2xl p-8 border-2 border-gray-200">
                <MessageCircle className="w-12 h-12 text-gray-900 mx-auto mb-4" />
                <h3 className="text-2xl font-bold text-gray-900 mb-2">
                  Still have questions?
                </h3>
                <p className="text-gray-600 mb-6">
                  Can't find the answer you're looking for? Our support team is
                  here to help.
                </p>
                <div className="flex gap-4 justify-center">
                  <Button
                    onClick={onGetStarted}
                    className="!bg-black hover:!bg-gray-800 !text-white !border-0 shadow-lg"
                  >
                    Contact Support
                  </Button>
                  <Button
                    variant="outline"
                    className="!border-2 !border-gray-800 !text-gray-900 hover:!bg-gray-100 !bg-white"
                  >
                    View Documentation
                  </Button>
                </div>
              </div>
            </motion.div>
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="max-w-7xl mx-auto px-8 py-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="rounded-3xl p-16 text-center relative overflow-hidden shadow-2xl border-2 border-gray-300"
          >
            {/* Background Image */}
            <div
              className="absolute inset-0 rounded-3xl"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1542744173-8e7e53415bb0?w=1920&h=1080&fit=crop')`,
                backgroundSize: "cover",
                backgroundPosition: "center",
              }}
            />

            {/* Dark overlay for text readability */}
            <div
              className="absolute inset-0 rounded-3xl"
              style={{
                background:
                  "linear-gradient(to bottom right, rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85))",
              }}
            />

            <div className="relative z-10">
              {/* Badge */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
                className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm border border-white/30 rounded-full px-6 py-2 mb-6"
              >
                <Sparkles className="w-5 h-5" style={{ color: "#fde047" }} />
                <span
                  className="font-semibold text-sm"
                  style={{ color: "#ffffff" }}
                >
                  Limited Time Offer
                </span>
              </motion.div>

              <h2
                className="text-5xl md:text-6xl mb-6 font-bold leading-tight"
                style={{ color: "#ffffff" }}
              >
                Ready to Transform Your Brand?
              </h2>
              <p
                className="text-lg md:text-xl mb-10 max-w-2xl mx-auto leading-relaxed"
                style={{ color: "#e5e7eb" }}
              >
                Join thousands of businesses using 1SYX to create powerful,
                memorable brands that drive results.
              </p>

              {/* CTA Buttons */}
              <div className="flex gap-4 justify-center mb-6">
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    onClick={onGetStarted}
                    className="bg-white text-black hover:bg-gray-100 border-0 shadow-xl rounded-xl px-10 py-6 text-lg font-semibold"
                  >
                    Get Started for Free
                    <ArrowRight className="ml-2 w-5 h-5" />
                  </Button>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    variant="outline"
                    className="!border-2 !border-white !text-white hover:!bg-white/10 !bg-transparent rounded-xl px-10 py-6 text-lg font-semibold"
                  >
                    Schedule a Demo
                  </Button>
                </motion.div>
              </div>

              {/* Trust indicators */}
              <div className="flex items-center justify-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle
                    className="w-5 h-5"
                    style={{ color: "#4ade80" }}
                  />
                  <span className="font-medium" style={{ color: "#e5e7eb" }}>
                    No credit card required
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle
                    className="w-5 h-5"
                    style={{ color: "#4ade80" }}
                  />
                  <span className="font-medium" style={{ color: "#e5e7eb" }}>
                    14-day free trial
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle
                    className="w-5 h-5"
                    style={{ color: "#4ade80" }}
                  />
                  <span className="font-medium" style={{ color: "#e5e7eb" }}>
                    Cancel anytime
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Footer */}
        <footer className="bg-gray-900 text-gray-300 py-12">
          <div className="max-w-7xl mx-auto px-8">
            <div className="grid grid-cols-4 gap-8 mb-8">
              <div>
                <div className="flex items-center gap-2 mb-4">
                  <img
                    src="/1syx-logo.jpeg"
                    alt="1SYX Logo"
                    className="h-8 w-auto"
                  />
                  <span className="text-xl font-semibold text-white">1SYX</span>
                </div>
                <p className="text-gray-400">1-System For Your 'X' Factor</p>
              </div>

              <div>
                <h4 className="text-white mb-4">Product</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Features
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Pricing
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Updates
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Company</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      About
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Blog
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Careers
                    </a>
                  </li>
                </ul>
              </div>

              <div>
                <h4 className="text-white mb-4">Legal</h4>
                <ul className="space-y-2">
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Privacy
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Terms
                    </a>
                  </li>
                  <li>
                    <a href="#" className="hover:text-white transition-colors">
                      Contact
                    </a>
                  </li>
                </ul>
              </div>
            </div>

            <div className="border-t border-gray-800 pt-8 flex justify-between items-center">
              <p className="text-gray-500">© 2024 1SYX. All rights reserved.</p>
              <div className="flex gap-4">
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Twitter
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  LinkedIn
                </a>
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition-colors"
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
