from enum import Enum
import transforms as T

################################################################################################################
##########          CLASS UTILS          #######################################################################
################################################################################################################

class Model_mode(Enum):
    resnet50 = 0
    mnasnet  = 1

################################################################################################################

class Distortion_mode(Enum):
    LOSSLESS           = 0
    JPEG               = 1
    JPEG2000           = 2
    BPG                = 3

################################################################################################################

class Distortion(object):
    def __init__(self, distortion_mode: int, quality = None, color_subsampling = None):
        if distortion_mode == Distortion_mode.LOSSLESS:
            self.DISTORTION_MODE             = Distortion_mode.LOSSLESS
            self.DISTORTION_STRING           = Distortion_mode.LOSSLESS.name
            self.QUALITY_SUPPORTED           = False
            self.COLOR_SUBSAMPLING_SUPPORTED = False
        elif distortion_mode == Distortion_mode.JPEG:
            self.DISTORTION_MODE             = Distortion_mode.JPEG
            self.DISTORTION_STRING           = Distortion_mode.JPEG.name
            self.QUALITY_SUPPORTED           = True
            self.COLOR_SUBSAMPLING_SUPPORTED = True
        elif distortion_mode == Distortion_mode.JPEG2000:
            self.DISTORTION_MODE             = Distortion_mode.JPEG2000
            self.DISTORTION_STRING           = Distortion_mode.JPEG2000.name
            self.QUALITY_SUPPORTED           = True
            self.COLOR_SUBSAMPLING_SUPPORTED = False
        elif distortion_mode == Distortion_mode.BPG:
            self.DISTORTION_MODE             = Distortion_mode.BPG
            self.DISTORTION_STRING           = Distortion_mode.BPG.name
            self.QUALITY_SUPPORTED           = True
            self.COLOR_SUBSAMPLING_SUPPORTED = True
        else:
            raise Exception("Unsupported distortion mode (" + str(distortion_mode) + ").")
        
        if quality is not None and not self.QUALITY_SUPPORTED:
            raise Exception("Quality not supported with distortion mode (" + str(distortion_mode) + "), but a quality parameter was provided.")
        if color_subsampling is not None and not self.COLOR_SUBSAMPLING_SUPPORTED:
            raise Exception("Color subsampling not supported with distortion mode (" + str(distortion_mode) + "), but a color_subsampling parameter was provided.")
        
        self.quality            = quality
        self.color_subsampling  = color_subsampling
    
    def get_compress_transform(self):
        if self.DISTORTION_MODE == Distortion_mode.LOSSLESS:
            return T.Identity()
        elif self.DISTORTION_MODE == Distortion_mode.JPEG:
            return T.JPEG(self.quality, color_subsampling=self.color_subsampling, p=1.0)
        elif self.DISTORTION_MODE == Distortion_mode.JPEG2000:
            return T.JPEG2000([self.quality], p=1.0)
        elif self.DISTORTION_MODE == Distortion_mode.BPG:
            return T.BPG(self.quality, color_subsampling=self.color_subsampling, p=1.0)
        else:
            raise Exception("Unsupported distortion mode (" + str(self.DISTORTION_MODE) + ").")
    
    def __str__(self):
        quality_string = ("quality=" + str(self.quality)) if self.QUALITY_SUPPORTED else ""
        if len(quality_string)>0 and self.COLOR_SUBSAMPLING_SUPPORTED:
            quality_string += ", "
        color_subsampling_string = ("color_subsampling=" + self.color_subsampling) if self.COLOR_SUBSAMPLING_SUPPORTED else ""
        return self.DISTORTION_STRING + "(" + quality_string + color_subsampling_string + ")"

################################################################################################################

class Distortion_precompute(Enum):
    SAVE_PRECOMPUTED_ENCODING           = -1 #Mode to save bitstream on disk (.jpeg, .j2k, .bpg)
    SAVE_PRECOMPUTED_ENCODING_DECODING  = -2 #Mode to save precomputed images on disk, saved losslessly (.ppm for very fast decoding, at the cost of disk storage)
    PRECOMPUTED_ON_THE_FLY              = 0  #Recommended for JPEG and JPEG2000 only. Mode to compute encoding/decoding on-the-fly (highest complexity, "no storage" space required) ("no storage" -> some codecs use disk space for temporary files).
    PRECOMPUTED_ENCODING                = 1  #Mode to load bitstream saved on disk (decoding only on-the-fly, "low" storage space required) ("low" -> depends on the codec/quality/etc. selected)
    PRECOMPUTED_ENCODING_DECODING       = 2  #Mode to load precomputed images saved on the disk losslessly (lowest complexity at training time, highest storage space required)

################################################################################################################
################################################################################################################
################################################################################################################