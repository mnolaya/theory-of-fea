import os
import itertools

from abaqus import mdb
import mesh
import abaqusConstants as abqconst

YDISP = 0.1
SEED = [2, 1, 0.5, 0.25, 0.125]
ORI = [0, 30, 45, 60, 90]
THICKNESS = 0.001
PROPS = {
    'E11': 231000,
    'E22': 15000,
    'E33': 15000,
    'nu12': 0.21,
    'nu13': 0.21,
    'nu23': 0.30,
    'G12': 15800,
    'G13': 15800,
    'G23': 5800,
}

# Create model
s = mdb.models['Model-1'].ConstrainedSketch(name='sketch', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(2.0, 10.0))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=abqconst.TWO_D_PLANAR, type=abqconst.DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# Create material
mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Elastic(
    type=abqconst.ENGINEERING_CONSTANTS, 
    table=(PROPS.values(), )
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
    u2=abqconst.UNSET, 
    ur3=abqconst.SET, 
    amplitude=abqconst.UNSET, 
    distributionType=abqconst.UNIFORM, 
    fieldName='', 
    localCsys=None
) # <- Top edge pinned in U1, UR12

mdb.models['Model-1'].EncastreBC(
    name='BC-2', 
    createStepName='Initial', 
    region=bottom_edge, 
    localCsys=None
) # <- Bottom edge fixed
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
    elemType1 = mesh.ElemType(elemCode=abqconst.CPS4, elemLibrary=abqconst.STANDARD)
    p.setElementType(regions=model_region, elemTypes=(elemType1, ))
    p.generateMesh()

    # Create material orientation
    ori = mdb.models['Model-1'].parts['Part-1'].MaterialOrientation(
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

    # Write the input file.
    job_name = 'proj_mesh{}_{}'.format(si+1, theta)
    if not os.path.exists(job_name): os.mkdir(job_name)
    os.chdir(job_name)
    job = mdb.Job(name=job_name, model='Model-1')
    job.writeInput()
    
    # Return to base working dir.
    os.chdir(base_work_dir)