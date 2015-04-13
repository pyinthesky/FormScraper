# @author github.com/pyinthesky
#
# running requirements (OSX)
# brew install tesseract
# python 3.4
# pip install pytesseract
# blanon@sunlightfoundation.org

import pprint
import re
import PIL
from PIL import Image
import pytesseract
import numpy as np
from scipy import misc

class Form990():
    """Uses image masks to perform OCR on the IRS Tax Form 990"""
    def __init__(self, image_path):
        self.image_path         = image_path
        self.tax_status         =   {
                                        "I0":"501(c)(3)",
                                        "I1":"501(c)(%s)",
                                        "I2":"4947(a)(1)",
                                        "I3":"527",
                                    }
        self.bounding_box_dict  =   {
            #                                 key: ( w-start, h-start, w-end, h-end )
            "form_omb":                                 (1025,   75, 1180,   90),
            "form_year":                                (1108,  100, 1170,  140),
            "box_a_tax_year":                           ( 100,  200, 1200,  221),
            "checkbox_b_address_change":                (  76,  253,   90,  267),
            "checkbox_b_name_change":                   (  76,  279,   90,  292),
            "checkbox_b_initial_return":                (  76,  303,   90,  319),
            "checkbox_b_final_return":                  (  76,  329,   90,  342),
            "checkbox_b_ammended_return":               (  76,  354,   90,  367),
            "checkbox_b_application_pending":           (  76,  379,   90,  392),
            "box_c_name_of_organization":               ( 400,  227,  927,  246),
            "box_c_doing_business_as":                  ( 380,  250,  927,  271),
            "box_c_address_street":                     ( 242,  294,  762,  320),
            "box_c_address_room_suite":                 ( 766,  294,  927,  321),
            "box_c_address_city_town_zip_postal_code":  ( 242,  344,  927,  370),
            "box_d_employer_identification_number_ein": ( 933,  244, 1200,  272),
            "box_e_telephone_number":                   ( 933,  291, 1200,  320),
            "box_g_gross_receipts":                     (1062,  325, 1200,  371),

            "checkbox_i_501c3":                         ( 270,  429,  284,  443),
            "checkbox_i_501cX":                         ( 420,  429,  434,  443),
            "checkbox_i_4947":                          ( 646,  429,  659,  443),
            "checkbox_i_527":                           ( 766,  429,  779,  443),




            "part1_box3":                               (1010,  625, 1200,  647),
            "part1_box4":                               (1010,  650, 1200,  672),
            "part1_box5":                               (1010,  675, 1200,  697),
            "part1_box6":                               (1010,  700, 1200,  722),
            "part1_box7a":                              (1010,  725, 1200,  747),
            "part1_box7b":                              (1010,  750, 1200,  772),

            "part1_box8_current_year":                  (1010,  800, 1200,  822),
            "part1_box9_current_year":                  (1010,  825, 1200,  847),
            "part1_box10_current_year":                 (1010,  850, 1200,  872),
            "part1_box11_current_year":                 (1010,  875, 1200,  897),
            "part1_box12_current_year":                 (1010,  900, 1200,  922),
            "part1_box13_current_year":                 (1010,  925, 1200,  942),
            "part1_box14_current_year":                 (1010,  950, 1200,  972),
            "part1_box15_current_year":                 (1010,  975, 1200,  997),
            "part1_box16a_current_year":                (1010, 1000, 1200, 1022),
            "part1_box16b_current_year":                (1010, 1025, 1200, 1047),
            "part1_box17_current_year":                 (1010, 1050, 1200, 1072),
            "part1_box18_current_year":                 (1010, 1075, 1200, 1097),
            "part1_box19_current_year":                 (1010, 1100, 1200, 1122),
            "part1_box20_current_year":                 (1010, 1150, 1200, 1172),
            "part1_box21_current_year":                 (1010, 1175, 1200, 1197),
            "part1_box22_current_year":                 (1010, 1200, 1200, 1222),

            "part1_box8_prior_year":                    ( 825,  800, 1000,  822),
            "part1_box9_prior_year":                    ( 825,  825, 1000,  847),
            "part1_box10_prior_year":                   ( 825,  850, 1000,  872),
            "part1_box11_prior_year":                   ( 825,  875, 1000,  897),
            "part1_box12_prior_year":                   ( 825,  900, 1000,  922),
            "part1_box13_prior_year":                   ( 825,  925, 1000,  942),
            "part1_box14_prior_year":                   ( 825,  950, 1000,  972),
            "part1_box15_prior_year":                   ( 825,  975, 1000,  997),
            "part1_box16a_prior_year":                  ( 825, 1000, 1000, 1022),
            "part1_box16b_prior_year":                  ( 825, 1025, 1000, 1047),
            "part1_box17_prior_year":                   ( 825, 1050, 1000, 1072),
            "part1_box18_prior_year":                   ( 825, 1075, 1000, 1097),
            "part1_box19_prior_year":                   ( 825, 1100, 1000, 1122),
            "part1_box20_prior_year":                   ( 825, 1150, 1000, 1172),
            "part1_box21_prior_year":                   ( 825, 1175, 1000, 1197),
            "part1_box22_prior_year":                   ( 825, 1200, 1000, 1222),
            "part2_printed_signature":                  ( 200, 1343, 1200, 1371),
        }

        self.component_contents_dict = dict(zip(self.bounding_box_dict.keys(), len(self.bounding_box_dict) * [""]))

    def parse(self):
        """runs each mask(crop) across the image file to improve OCR functionality"""
        image = Image.open(self.image_path)
        for form_field, bounding_box in self.bounding_box_dict.items():
            # the crops are scaled up and the contrast maxed out in order to enhance character
            # features and increase OCR success
            x1, y1, x2, y2  = bounding_box
            xx              = (x2-x1) << 2
            yy              = (y2-y1) << 2
            the_crop        = image.crop(bounding_box)
            the_crop        = the_crop.resize((xx,yy),PIL.Image.LANCZOS)
            area            = (xx * yy)
            gray            = the_crop.convert('L')
            bw              = np.asarray(gray).copy()
            bw[bw  < 200]   = 0
            bw[bw >= 200]   = 255
            the_crop        = misc.toimage(bw)

            # use this to check out a particular mask
            #if "box_c_address_city_town_zip_postal_code" is form_field:
            #    the_crop.show()

            if "checkbox" in form_field:
                # a box is considered checked if 10% or more of it's area is black
                checked = np.sum(bw) >= (0.1 * area)
                self.component_contents_dict[form_field] = checked
            else:
                self.component_contents_dict[form_field] = self.clean_text(pytesseract.image_to_string(the_crop))
        print([self.component_contents_dict['box_c_address_city_town_zip_postal_code']])

    def clean_text(self, st):
        """character cleanup for common/repeatable OCR problems"""
        st = re.sub('‘!', '1', st)
        st = re.sub(r'(\d) (\d)', r'\1\2', st)
        st = re.sub(r'\n|\r',' ', st)
        return st

    def __repr__(self):
        """returns the pretty formatted version of the image data contents"""
        return pprint.pformat(self.component_contents_dict)

    @classmethod
    def edges(cls):
        from scipy import ndimage, misc
        import numpy as np
        from skimage import feature
        col = Image.open("f990.jpg")
        gray = col.convert('L')

        # Let numpy do the heavy lifting for converting pixels to pure black or white
        bw = np.asarray(gray).copy()

        # Pixel range is 0...255, 256/2 = 128
        bw[bw < 245]  = 0    # Black
        bw[bw >= 245] = 255 # White
        bw[bw == 0] = 254
        bw[bw == 255] = 0
        im = bw
        im = ndimage.gaussian_filter(im, 1)
        edges2 = feature.canny(im, sigma=2)
        labels, numobjects =ndimage.label(im)
        slices = ndimage.find_objects(labels)
        print('\n'.join(map(str, slices)))
        misc.imsave('f990_sob.jpg', im)
        return

        #im = misc.imread('f990.jpg')
        #im = ndimage.gaussian_filter(im, 8)
        sx = ndimage.sobel(im, axis=0, mode='constant')
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        misc.imsave('f990_sob.jpg', edges2)

if __name__ == "__main__":
    f = Form990('f990_a.jpg')
    f.parse()
    print(f)
