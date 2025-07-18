class SnailObject:
    """
    Represents a single detected snail with its measurements and geometry.
    """
    def __init__(self, snail_id, length, width, contour, bounding_box):
        """
        Args:
            snail_id (str or int): Unique identifier for the snail.
            length (float): Length of the snail in mm.
            width (float): Width of the snail in mm.
            contour (np.ndarray): Contour points for the snail.
            bounding_box (np.ndarray): 4x2 array of bounding box coordinates.
        """
        self.snail_id = snail_id
        self.length = length
        self.width = width
        self.contour = contour
        self.bounding_box = bounding_box

    def to_dict(self):
        """
        Returns a dictionary representation of the snail object.
        """
        return {
            "id": self.snail_id,
            "length_mm": self.length,
            "width_mm": self.width,
            "bounding_box": self.bounding_box.tolist() if hasattr(self.bounding_box, "tolist") else self.bounding_box,
            "contour": self.contour.tolist() if hasattr(self.contour, "tolist") else self.contour
        }

    def __repr__(self):
        return f"SnailObject(id={self.snail_id}, length={self.length:.2f}mm"