    max_expansion = min(len(list(self.state.legal_moves)), int(math.sqrt(self.visit)) + 1)
        return len(self.childs) >= max_expansion
