from django.contrib import admin

# Register your models here.
from . models import Product, Supplies, Orders, OrderUpdate

admin.site.register(Product)
admin.site.register(Supplies) 
admin.site.register(Orders) 
admin.site.register(OrderUpdate) 

