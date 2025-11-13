import { CheckCircle2, Circle, Sparkles } from 'lucide-react';
import { Button } from './ui/button';
import { motion } from 'motion/react';

export function ScoreSection() {
  const sections = [
    { name: 'What Use?', completed: true },
    { name: 'Brand Info & Resources', completed: true },
    { name: 'Score', completed: false },
  ];

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
        <h2 className="text-center relative z-10">Review & Score</h2>
      </motion.div>
      
      <div className="grid grid-cols-2 gap-8 mb-8">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-white border-2 border-indigo-200 rounded-2xl p-8 shadow-lg"
        >
          <h3 className="mb-6 text-indigo-700">Completion Status</h3>
          <div className="space-y-4">
            {sections.map((section, index) => (
              <motion.div 
                key={section.name} 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-indigo-50 transition-colors duration-200"
              >
                {section.completed ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 500, delay: 0.5 + index * 0.1 }}
                  >
                    <CheckCircle2 className="w-6 h-6 text-green-500" />
                  </motion.div>
                ) : (
                  <Circle className="w-6 h-6 text-gray-400" />
                )}
                <span>{section.name}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
        
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-2xl p-8 shadow-lg"
        >
          <h3 className="mb-4 text-indigo-700">Your Score</h3>
          <div className="text-center py-8">
            <motion.div 
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ 
                type: "spring", 
                stiffness: 200, 
                damping: 15,
                delay: 0.5 
              }}
              className="bg-gradient-to-br from-indigo-500 to-purple-500 text-white rounded-2xl inline-block px-12 py-8 mb-4 shadow-xl"
            >
              <motion.span 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
                className="text-6xl"
              >
                85
              </motion.span>
              <span className="text-3xl">/100</span>
            </motion.div>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              className="text-gray-600"
            >
              Great progress! Complete all sections to improve your score.
            </motion.p>
          </div>
        </motion.div>
      </div>
      
      <div className="flex justify-end">
        <motion.div 
          whileHover={{ scale: 1.05 }} 
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.2 }}
        >
          <Button className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white rounded-xl shadow-lg border-0 relative overflow-hidden">
            <div 
              className="absolute inset-0 opacity-30 bg-cover bg-center"
              style={{
                backgroundImage: `url('https://images.unsplash.com/photo-1646038572891-86b08ccd6719?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwd2F2ZXN8ZW58MXx8fHwxNzYzMDExMDc0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral')`,
              }}
            />
            <span className="relative z-10 flex items-center">
              <Sparkles className="mr-2 w-4 h-4" />
              Let's Generate
            </span>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}