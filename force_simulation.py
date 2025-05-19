class TearSimulator:
    def __init__(self, stiffness=5.0, tear_threshold=30.0):
        self.stiffness = stiffness  # N/mm (simplified)
        self.tear_threshold = tear_threshold  # Breaking force (N)
        self.deformation = 0.0
        self.force = 0.0
        self.torn = False
        self.progress = 0.0  # % of tear severity
        self.set_profile(stiffness, tear_threshold)
        self.reset()

    def apply_displacement(self, displacement_mm):
        self.deformation = displacement_mm
        self.force = self.stiffness * self.deformation
        self.torn = self.force >= self.tear_threshold
        self.progress = min(100, (self.force / self.tear_threshold) * 100)
        return self.force, self.torn, self.progress

    def reset(self):
        self.force = 0.0
        self.deformation = 0.0
        self.torn = False
        self.progress = 0.0

    def set_profile(self, stiffness, tear_threshold):
        self.stiffness = stiffness
        self.tear_threshold = tear_threshold
