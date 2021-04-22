python localize_text_tesseract.py
param:
-i image path
-l language: chi-sim/eng/deu/fra which means Chinese-simple/English/German/French
-c mininum confidence

notes:
use u"好的不客气" to print unicode
cv2.puttext cannot print unicode now!!!