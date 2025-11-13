import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { ArrowRight, Upload } from 'lucide-react';
import { motion } from 'motion/react';

interface BrandInfoSectionProps {
  onNext: () => void;
}

export function BrandInfoSection({ onNext }: BrandInfoSectionProps) {
  return (
    <div className="max-w-6xl">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-2xl p-12 mb-8 shadow-xl relative overflow-hidden"
      >
        <div 
          className="absolute inset-0 opacity-20 bg-cover bg-center"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
          }}
        />
        <h2 className="text-center relative z-10">Brand Information & Resources</h2>
      </motion.div>
      
      <div className="grid grid-cols-2 gap-8">
        {/* Left Column - Brand Info */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6 bg-white rounded-2xl p-8 shadow-lg"
        >
          <h3 className="text-indigo-700 mb-6">Basic Information</h3>
          
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Label htmlFor="brand-name" className="text-indigo-700">Brand Name</Label>
            <Input 
              id="brand-name" 
              className="border-2 border-indigo-200 rounded-xl mt-2 focus:border-indigo-400 focus:ring-indigo-400 transition-all duration-200"
              placeholder="Enter your brand name"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <Label htmlFor="industry" className="text-indigo-700">Industry</Label>
            <Input 
              id="industry" 
              className="border-2 border-indigo-200 rounded-xl mt-2 focus:border-indigo-400 focus:ring-indigo-400 transition-all duration-200"
              placeholder="e.g., Technology, Fashion, Food"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.5 }}
          >
            <Label htmlFor="description" className="text-indigo-700">Brand Description</Label>
            <Textarea 
              id="description" 
              className="border-2 border-indigo-200 rounded-xl mt-2 min-h-[120px] focus:border-indigo-400 focus:ring-indigo-400 transition-all duration-200"
              placeholder="Tell us about your brand..."
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.6 }}
          >
            <Label htmlFor="target-audience" className="text-indigo-700">Target Audience</Label>
            <Input 
              id="target-audience" 
              className="border-2 border-indigo-200 rounded-xl mt-2 focus:border-indigo-400 focus:ring-indigo-400 transition-all duration-200"
              placeholder="Who is your target audience?"
            />
          </motion.div>
        </motion.div>

        {/* Right Column - Resources */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <div className="bg-white rounded-2xl p-8 shadow-lg">
            <h3 className="text-indigo-700 mb-6">Resources</h3>
            
            <div className="space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
              >
                <Label className="text-indigo-700">Brand Assets</Label>
                <motion.div 
                  whileHover={{ scale: 1.02 }}
                  className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-2xl p-6 mt-2 text-center hover:border-indigo-400 hover:shadow-lg transition-all duration-300"
                >
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <Upload className="w-10 h-10 mx-auto mb-3 text-indigo-500" />
                  </motion.div>
                  <p className="mb-3 text-gray-600 text-sm">Upload logos, images, fonts</p>
                  <Button variant="outline" className="border-2 border-indigo-300 text-indigo-600 hover:bg-indigo-50 rounded-xl transition-all duration-200">
                    Choose Files
                  </Button>
                </motion.div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.4 }}
              >
                <Label className="text-indigo-700">Existing Materials</Label>
                <motion.div 
                  whileHover={{ scale: 1.02 }}
                  className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-2xl p-6 mt-2 text-center hover:border-indigo-400 hover:shadow-lg transition-all duration-300"
                >
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <Upload className="w-10 h-10 mx-auto mb-3 text-purple-500" />
                  </motion.div>
                  <p className="mb-3 text-gray-600 text-sm">Upload marketing materials</p>
                  <Button variant="outline" className="border-2 border-purple-300 text-purple-600 hover:bg-purple-50 rounded-xl transition-all duration-200">
                    Choose Files
                  </Button>
                </motion.div>
              </motion.div>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
              >
                <Label className="text-indigo-700">Reference Links</Label>
                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-2xl p-6 mt-2">
                  <input
                    type="url"
                    placeholder="https://example.com"
                    className="w-full border-2 border-indigo-200 rounded-xl p-3 mb-3 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200 transition-all duration-200"
                  />
                  <Button variant="outline" className="border-2 border-indigo-300 text-indigo-600 hover:bg-indigo-50 rounded-xl w-full transition-all duration-200">
                    Add Link
                  </Button>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
      
      <div className="flex justify-end mt-8">
        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
          <Button 
            onClick={onNext}
            className="bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 text-white rounded-xl shadow-lg border-0 relative overflow-hidden"
          >
            <div 
              className="absolute inset-0 opacity-30 bg-cover bg-center"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
              }}
            />
            <span className="relative z-10 flex items-center">
              Next <ArrowRight className="ml-2 w-4 h-4" />
            </span>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}