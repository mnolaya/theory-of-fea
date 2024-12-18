import os
import itertools

from abaqus import mdb
import mesh
import abaqusConstants as abqconst

YDISP = 0.1
SEED = [2, 1, 0.5, 0.25, 0.125, 1/16.0]
ORI = [0, 30, 45, 60, 90]
THICKNESS = 0.001
PROPS = [
    231000,
    15000,
    15000,
    0.21,
    0.21,
    0.30,
    15800,
    15800,
    5800,
]
ELEM_TYPE = 'linear'

# Create model
s = mdb.models['Model-1'].ConstrainedSketch(name='sketch', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(2.0, 10.0))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=abqconst.TWO_D_PLANAR, type=abqconst.DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# Create material
mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Elastic(
    type=abqconst.ENGINEERING_CONSTANTS, 
    table=(PROPS, )
)
model_region = p.Set(faces=p.faces, name='all')
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=THICKNESS)
p.SectionAssignment(region=model_region, sectionName='Section-1', offset=0.0, 
    offsetType=abqconst.MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=abqconst.FROM_SECTION)

# Create step
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial')

# Create assembly
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(abqconst.CARTESIAN)
a.Instance(name='Part-1-1', part=p, dependent=abqconst.ON)

# Create sets for BCs
edges = a.instances['Part-1-1'].edges
bottom_edge = a.Set(edges=edges.getSequenceFromMask(mask=('[#1 ]', ), ), name='bottom')
top_edge = a.Set(edges=edges.getSequenceFromMask(mask=('[#4 ]', ), ), name='top')

# Set BCs
mdb.models['Model-1'].DisplacementBC(
    name='BC-1', 
    createStepName='Initial', 
    region=top_edge, 
    u1=abqconst.SET, 
    u2=abqconst.SET, 
    ur3=abqconst.UNSET, 
    amplitude=abqconst.UNSET, 
    distributionType=abqconst.UNIFORM, 
    fieldName='', 
    localCsys=None
) # <- Top edge pinned in U1, U2

mdb.models['Model-1'].DisplacementBC(
    name='BC-2', 
    createStepName='Initial', 
    region=bottom_edge, 
    u1=abqconst.SET, 
    u2=abqconst.SET, 
    ur3=abqconst.UNSET, 
    amplitude=abqconst.UNSET, 
    distributionType=abqconst.UNIFORM, 
    fieldName='', 
    localCsys=None
) # <- Bottom edge pinned in U1, U2

mdb.models['Model-1'].DisplacementBC(
    name='BC-3', 
    createStepName='Step-1',
    region=top_edge,
    u1=abqconst.UNSET,
    u2=YDISP,
    ur3=abqconst.UNSET,
    amplitude=abqconst.UNSET,
    fixed=abqconst.OFF,
    distributionType=abqconst.UNIFORM,
    fieldName='',
    localCsys=None
) # <- Top edge displacement in Y

base_work_dir = os.getcwd()
for si, theta in itertools.product(range(len(SEED)), ORI):
    # Mesh part
    p.seedPart(size=SEED[si], deviationFactor=0.1, minSizeFactor=0.1)
    if ELEM_TYPE == 'linear':
        et = abqconst.CPS4
    else:
        et = abqconst.CPS8
    elemType1 = mesh.ElemType(elemCode=et, elemLibrary=abqconst.STANDARD)
    p.setElementType(regions=model_region, elemTypes=(elemType1, ))
    p.generateMesh()

    # Create material orientation
    ori = p.MaterialOrientation(
        region=model_region,
        orientationType=abqconst.SYSTEM,
        axis=abqconst.AXIS_3,
        localCsys=None,
        fieldName='',
        additionalRotationType=abqconst.ROTATION_ANGLE,
        additionalRotationField='',
        angle=theta,
        stackDirection=abqconst.STACK_3
    )

    # Create set for averaging
    gage_elems = p.elements.getByBoundingBox(xMin=0, xMax=2, yMin=2, yMax=8)
    p.Set(elements=gage_elems, name='gage')

    # Write the input file.
    job_name = 'proj_mesh{}_{}'.format(si+1, theta)
    job_dir = os.path.join(os.getcwd(), ELEM_TYPE, 'proj_mesh{}_{}'.format(si+1, theta))
    if not os.path.exists(job_dir): os.mkdir(job_dir)
    os.chdir(job_dir)
    job = mdb.Job(name=job_name, model='Model-1')
    job.writeInput()
    
    # Return to base working dir.
    os.chdir(base_work_dir)