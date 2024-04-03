from diving_analysis.models import OriginalPredictedBoundingBox, EnhancedPredictedBoundingBox


class BoundingBoxDivingFyClass:
    DIVER = OriginalPredictedBoundingBox.BOX_CLASS[0][1]
    SPRINGBOARD = OriginalPredictedBoundingBox.BOX_CLASS[1][1]
    WATER_SLASH = OriginalPredictedBoundingBox.BOX_CLASS[2][1]
    TOP_SPRINGBOARD = EnhancedPredictedBoundingBox.ENHANCED_BOX_CLASS[2][1]
    BOTTOM_SPRINGBOARD = EnhancedPredictedBoundingBox.ENHANCED_BOX_CLASS[3][1]

    @staticmethod
    def get_code(bbox_class):
        if bbox_class == BoundingBoxDivingFyClass.DIVER:
            return OriginalPredictedBoundingBox.BOX_CLASS[0][0]
        elif bbox_class == BoundingBoxDivingFyClass.SPRINGBOARD:
            return OriginalPredictedBoundingBox.BOX_CLASS[1][0]
        elif bbox_class == BoundingBoxDivingFyClass.WATER_SLASH:
            return OriginalPredictedBoundingBox.BOX_CLASS[2][0]
        elif bbox_class == BoundingBoxDivingFyClass.TOP_SPRINGBOARD:
            return EnhancedPredictedBoundingBox.ENHANCED_BOX_CLASS[2][0]
        elif bbox_class == BoundingBoxDivingFyClass.BOTTOM_SPRINGBOARD:
            return EnhancedPredictedBoundingBox.ENHANCED_BOX_CLASS[3][0]
        else:
            return None
