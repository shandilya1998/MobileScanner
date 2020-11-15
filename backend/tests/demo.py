from reportlab.pdfgen import canvas    
from reportlab.lib.units import inch

def hello(c):
    c.translate(inch,inch)
    c.setFont("Helvetica", 14)
    c.setStrokeColorRGB(0.2,0.5,0.3)
    c.setFillColorRGB(1,0,1)
    c.line(0,0,0,1.7*inch)
    c.line(0,0,1*inch,0)
    c.rect(0.2*inch,0.2*inch,1*inch,1.5*inch, fill=1)
    c.rotate(90)
    c.setFillColorRGB(0,0,0.77)
    c.drawString(0.3*inch, -inch, "Hello World")
c = canvas.Canvas("hello.pdf")    
hello(c)    
c.showPage()    
c.save()
