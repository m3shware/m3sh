{{ name | escape | underline }}

.. currentmodule:: {{ module }}
 
.. autoclass:: {{ objname }}
   :show-inheritance:
   :no-inherited-members:
    
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
 
   .. autosummary::
      :toctree:
      :template: attribute.rst
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
 
   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
 
   .. autosummary::
      :toctree: 
      :template: function.rst
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   
